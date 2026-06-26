#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Constants.h"
#include <vector>
#include <map>

using namespace llvm;

namespace {

// -----------------------------------------------------------------------
// Type helpers
// -----------------------------------------------------------------------

StructType* getDoubleSingleType(LLVMContext& Ctx) {
    return StructType::getTypeByName(Ctx, "DoubleSingle") ?:
           StructType::create(Ctx, {Type::getFloatTy(Ctx), Type::getFloatTy(Ctx)}, "DoubleSingle");
}

// Recursively replace every double leaf with dsType.
Type* transformType(Type* T, StructType* dsType) {
    if (T->isDoubleTy())
        return dsType;
    if (auto* arr = dyn_cast<ArrayType>(T))
        return ArrayType::get(transformType(arr->getElementType(), dsType), arr->getNumElements());
    return T;
}

// Return true if T contains a double anywhere.
bool containsDouble(Type* T) {
    if (T->isDoubleTy()) return true;
    if (auto* arr = dyn_cast<ArrayType>(T))
        return containsDouble(arr->getElementType());
    return false;
}

// -----------------------------------------------------------------------
// Pass
// -----------------------------------------------------------------------

struct SkeletonPass : public PassInfoMixin<SkeletonPass> {
    StructType*  dsType = nullptr;
    LLVMContext* ctx    = nullptr;
    Module*      mod    = nullptr;

    // Runtime function handles
    FunctionCallee convertToDSFunc;   // void(DS*, double)
    FunctionCallee dsToDoubleFunc;    // double(DS*)
    FunctionCallee addFunc, subFunc, mulFunc, divFunc; // void(DS*, DS*, DS*)
    FunctionCallee negFunc, sqrtFunc, fabsFunc;        // void(DS*, DS*)
    FunctionCallee fmaddFunc;         // void(DS* result, DS* a, DS* b, DS* c)
    FunctionCallee printDSFunc;       // void(DS)

    // Per-function state
    std::map<Value*, Value*> valueMap;    // original value -> DS replacement
    std::map<Value*, Type*>  origTypeMap; // original alloca -> original double type
    std::vector<Instruction*> toDelete;
    std::vector<PHINode*>     phisToFix;

    // Per-function scratch alloca slots
    AllocaInst* tempOp1;
    AllocaInst* tempOp2;
    AllocaInst* tempOp3;    // separate slot for fmuladd c-operand (avoids aliasing tempResult)
    AllocaInst* tempResult;

    // ----------------------------------------------------------------
    void declareExternalFunctions(Module& M) {
        auto voidTy   = Type::getVoidTy(*ctx);
        auto dsPtrTy  = PointerType::getUnqual(dsType);
        auto doubleTy = Type::getDoubleTy(*ctx);

        convertToDSFunc = M.getOrInsertFunction("convert_double_to_ds",
            FunctionType::get(voidTy, {dsPtrTy, doubleTy}, false));

        dsToDoubleFunc = M.getOrInsertFunction("ds_to_double",
            FunctionType::get(doubleTy, {dsPtrTy}, false));

        fmaddFunc = M.getOrInsertFunction("external_double_fmadd",
            FunctionType::get(voidTy, {dsPtrTy, dsPtrTy, dsPtrTy, dsPtrTy}, false));

        auto binTy = FunctionType::get(voidTy, {dsPtrTy, dsPtrTy, dsPtrTy}, false);
        addFunc = M.getOrInsertFunction("external_double_add", binTy);
        subFunc = M.getOrInsertFunction("external_double_sub", binTy);
        mulFunc = M.getOrInsertFunction("external_double_mul", binTy);
        divFunc = M.getOrInsertFunction("external_double_div", binTy);

        auto unaTy = FunctionType::get(voidTy, {dsPtrTy, dsPtrTy}, false);
        negFunc  = M.getOrInsertFunction("external_double_neg",  unaTy);
        sqrtFunc = M.getOrInsertFunction("external_double_sqrt", unaTy);
        fabsFunc = M.getOrInsertFunction("external_double_fabs", unaTy);

        printDSFunc = M.getOrInsertFunction("print_ds_value",
            FunctionType::get(voidTy, {dsType}, false));
    }

    void createTemporaryStorage(Function& F) {
        IRBuilder<> B(&F.getEntryBlock(), F.getEntryBlock().begin());
        tempOp1    = B.CreateAlloca(dsType, nullptr, "ds.op1");
        tempOp2    = B.CreateAlloca(dsType, nullptr, "ds.op2");
        tempOp3    = B.CreateAlloca(dsType, nullptr, "ds.op3");
        tempResult = B.CreateAlloca(dsType, nullptr, "ds.result");
        for (auto* a : {tempOp1, tempOp2, tempOp3, tempResult})
            a->setAlignment(Align(8));
    }

    // ----------------------------------------------------------------
    // Return DS value for v.  Handles:
    //   - values already in valueMap
    //   - double ConstantFP (convert on the fly into tempStorage)
    //   - double undef
    Value* toDS(IRBuilder<>& B, Value* v, AllocaInst* tempStorage) {
        if (valueMap.count(v))
            return valueMap[v];
        if (isa<UndefValue>(v) && v->getType()->isDoubleTy())
            return UndefValue::get(dsType);
        if (auto* cfp = dyn_cast<ConstantFP>(v)) {
            if (cfp->getType()->isDoubleTy()) {
                B.CreateCall(convertToDSFunc, {tempStorage, v});
                return B.CreateLoad(dsType, tempStorage);
            }
        }
        return nullptr;
    }

    // Convert a DS value back to double (store then call ds_to_double).
    Value* fromDS(IRBuilder<>& B, Value* dsVal, AllocaInst* tempStorage) {
        B.CreateStore(dsVal, tempStorage);
        return B.CreateCall(dsToDoubleFunc, {tempStorage});
    }

    // ----------------------------------------------------------------
    // Recursive helpers for array init (works for any dimensionality / size)
    // indices starts as {0} and grows with each array dimension walked.

    void emitConvertElements(IRBuilder<>& B,
                              Type* curSrcTy, Type* baseSrcTy, Type* baseDstTy,
                              Value* srcBase, Value* dstBase,
                              SmallVector<Value*, 4>& idx) {
        if (curSrcTy->isDoubleTy()) {
            Value* srcGEP = B.CreateInBoundsGEP(baseSrcTy, srcBase, idx);
            Value* dbl    = B.CreateLoad(Type::getDoubleTy(*ctx), srcGEP);
            Value* dstGEP = B.CreateInBoundsGEP(baseDstTy, dstBase, idx);
            B.CreateCall(convertToDSFunc, {dstGEP, dbl});
            return;
        }
        if (auto* arr = dyn_cast<ArrayType>(curSrcTy)) {
            for (uint64_t i = 0; i < arr->getNumElements(); i++) {
                idx.push_back(ConstantInt::get(Type::getInt32Ty(*ctx), i));
                emitConvertElements(B, arr->getElementType(), baseSrcTy, baseDstTy,
                                    srcBase, dstBase, idx);
                idx.pop_back();
            }
        }
    }

    void emitZeroElements(IRBuilder<>& B,
                           Type* curTy, Type* baseTy,
                           Value* dstBase, SmallVector<Value*, 4>& idx,
                           Constant* zeroDS) {
        if (curTy == dsType) {
            B.CreateStore(zeroDS, B.CreateInBoundsGEP(baseTy, dstBase, idx));
            return;
        }
        if (auto* arr = dyn_cast<ArrayType>(curTy)) {
            for (uint64_t i = 0; i < arr->getNumElements(); i++) {
                idx.push_back(ConstantInt::get(Type::getInt32Ty(*ctx), i));
                emitZeroElements(B, arr->getElementType(), baseTy, dstBase, idx, zeroDS);
                idx.pop_back();
            }
        }
    }

    // ----------------------------------------------------------------
    // Handle binary FP operations (fadd/fsub/fmul/fdiv)
    void handleBinOp(BinaryOperator* I, FunctionCallee fn) {
        IRBuilder<> B(I);
        Value* lDS = toDS(B, I->getOperand(0), tempOp1);
        Value* rDS = toDS(B, I->getOperand(1), tempOp2);
        if (!lDS || !rDS) return;
        B.CreateStore(lDS, tempOp1);
        B.CreateStore(rDS, tempOp2);
        B.CreateCall(fn, {tempResult, tempOp1, tempOp2});
        valueMap[I] = B.CreateLoad(dsType, tempResult);
        toDelete.push_back(I);
    }

    // Handle unary FP operations (fneg, sqrt, fabs)
    void handleUnaryOp(Instruction* I, Value* operand, FunctionCallee fn) {
        IRBuilder<> B(I);
        Value* opDS = toDS(B, operand, tempOp1);
        if (!opDS) return;
        B.CreateStore(opDS, tempOp1);
        B.CreateCall(fn, {tempResult, tempOp1});
        valueMap[I] = B.CreateLoad(dsType, tempResult);
        toDelete.push_back(I);
    }

    // Handle memcpy / memset on double arrays
    void handleArrayInit(CallInst* call, Value* dst, Value* src, bool isZero) {
        if (!valueMap.count(dst) || !origTypeMap.count(dst)) return;
        IRBuilder<> B(call);
        Value* newDst     = valueMap[dst];
        Type*  origTy     = origTypeMap[dst];
        Type*  newTy      = cast<AllocaInst>(newDst)->getAllocatedType();
        SmallVector<Value*, 4> idx = {ConstantInt::get(Type::getInt32Ty(*ctx), 0)};
        if (!isZero && src) {
            emitConvertElements(B, origTy, origTy, newTy, src, newDst, idx);
        } else if (isZero) {
            Constant* z0  = ConstantFP::get(Type::getFloatTy(*ctx), 0.0);
            Constant* zDS = ConstantStruct::get(dsType, {z0, z0});
            emitZeroElements(B, newTy, newTy, newDst, idx, zDS);
        }
        toDelete.push_back(call);
    }

    // Handle calls to external (non-intrinsic) functions that take/return doubles.
    // Converts DS args back to double, then converts a double return back to DS.
    void handleExternalCall(CallInst* call) {
        Function* fn = call->getCalledFunction();
        if (!fn || fn->isIntrinsic()) return;
        // Skip our own runtime functions — they must not be re-processed
        static const char* runtimeNames[] = {
            "convert_double_to_ds", "ds_to_double", "print_ds_value",
            "external_double_add",  "external_double_sub", "external_double_mul",
            "external_double_div",  "external_double_neg", "external_double_sqrt",
            "external_double_fabs", "external_double_fmadd"
        };
        for (auto* n : runtimeNames)
            if (fn->getName() == n) return;
        // Only act if at least one double arg is in valueMap, or return is double
        bool needsTransform = false;
        for (unsigned i = 0; i < call->arg_size(); i++) {
            if (call->getArgOperand(i)->getType()->isDoubleTy() &&
                valueMap.count(call->getArgOperand(i)))
                needsTransform = true;
        }
        if (!needsTransform && !call->getType()->isDoubleTy()) return;

        IRBuilder<> B(call);
        SmallVector<Value*, 8> newArgs;
        for (unsigned i = 0; i < call->arg_size(); i++) {
            Value* arg = call->getArgOperand(i);
            if (arg->getType()->isDoubleTy() && valueMap.count(arg)) {
                // Need a dedicated temp per arg to avoid clobbering
                // Use tempOp1/tempOp2 for the first two; for more, reuse tempOp2
                AllocaInst* tmp = (i == 0) ? tempOp1 : tempOp2;
                newArgs.push_back(fromDS(B, valueMap[arg], tmp));
            } else {
                newArgs.push_back(arg);
            }
        }

        // Build the new call (void calls must not be named in LLVM IR)
        Twine callName = call->getType()->isVoidTy() ? Twine() : Twine(call->getName() + ".dbl");
        CallInst* newCall = B.CreateCall(fn->getFunctionType(), fn, newArgs, callName);

        if (call->getType()->isDoubleTy()) {
            // Wrap the returned double in DS
            B.CreateCall(convertToDSFunc, {tempResult, newCall});
            valueMap[call] = B.CreateLoad(dsType, tempResult);
        } else {
            call->replaceAllUsesWith(newCall);
        }
        toDelete.push_back(call);
    }

    // ----------------------------------------------------------------
    void processInstruction(Instruction* I) {

        // --- alloca: replace double (or double-array) allocas ---
        if (auto* A = dyn_cast<AllocaInst>(I)) {
            Type* ty = A->getAllocatedType();
            if (!containsDouble(ty)) return;
            IRBuilder<> B(A);
            Type* newTy   = transformType(ty, dsType);
            auto* newA    = B.CreateAlloca(newTy, nullptr, A->getName() + ".ds");
            newA->setAlignment(Align(8));
            valueMap[A]    = newA;
            origTypeMap[A] = ty;
            toDelete.push_back(A);
            return;
        }

        // --- load: load DS value ---
        if (auto* L = dyn_cast<LoadInst>(I)) {
            if (!valueMap.count(L->getPointerOperand())) return;
            IRBuilder<> B(L);
            Value* ptr  = valueMap[L->getPointerOperand()];
            Type*  ldTy = dsType;
            if (auto* ap = dyn_cast<AllocaInst>(ptr)) ldTy = ap->getAllocatedType();
            valueMap[L] = B.CreateLoad(ldTy, ptr, L->getName() + ".ds");
            toDelete.push_back(L);
            return;
        }

        // --- store: store DS value or convert double constant ---
        if (auto* S = dyn_cast<StoreInst>(I)) {
            if (!valueMap.count(S->getPointerOperand())) return;
            IRBuilder<> B(S);
            Value* val = S->getValueOperand();
            Value* dst = valueMap[S->getPointerOperand()];
            if (valueMap.count(val)) {
                B.CreateStore(valueMap[val], dst);
                toDelete.push_back(S);
            } else if (val->getType()->isDoubleTy()) {
                // Direct double constant/argument -> convert into the DS slot
                B.CreateCall(convertToDSFunc, {dst, val});
                toDelete.push_back(S);
            }
            return;
        }

        // --- bitcast: in typed-pointer IR, allocas are cast to i8* for memcpy/memset.
        // Mark such casts for deletion since their alloca is being replaced.
        if (auto* BC = dyn_cast<BitCastInst>(I)) {
            if (BC->getType()->isPointerTy() && valueMap.count(BC->getOperand(0)))
                toDelete.push_back(BC);
            return;
        }

        // --- getelementptr: remap to DS array ---
        if (auto* G = dyn_cast<GetElementPtrInst>(I)) {
            if (!valueMap.count(G->getPointerOperand())) return;
            IRBuilder<> B(G);
            Value* newBase = valueMap[G->getPointerOperand()];
            SmallVector<Value*, 4> idx(G->idx_begin(), G->idx_end());
            Type* baseTy = (isa<AllocaInst>(newBase))
                ? cast<AllocaInst>(newBase)->getAllocatedType()
                : transformType(G->getSourceElementType(), dsType);
            valueMap[G] = B.CreateInBoundsGEP(baseTy, newBase, idx, G->getName() + ".ds");
            toDelete.push_back(G);
            return;
        }

        // --- binary FP arithmetic ---
        if (auto* BO = dyn_cast<BinaryOperator>(I)) {
            if (!BO->getType()->isDoubleTy()) return;
            switch (BO->getOpcode()) {
                case Instruction::FAdd: handleBinOp(BO, addFunc); break;
                case Instruction::FSub: handleBinOp(BO, subFunc); break;
                case Instruction::FMul: handleBinOp(BO, mulFunc); break;
                case Instruction::FDiv: handleBinOp(BO, divFunc); break;
                default: break;
            }
            return;
        }

        // --- unary FP (fneg) ---
        if (auto* UO = dyn_cast<UnaryOperator>(I)) {
            if (!UO->getType()->isDoubleTy()) return;
            if (UO->getOpcode() == Instruction::FNeg)
                handleUnaryOp(UO, UO->getOperand(0), negFunc);
            return;
        }

        // --- select (conditional select between two doubles) ---
        if (auto* SEL = dyn_cast<SelectInst>(I)) {
            if (!SEL->getType()->isDoubleTy()) return;
            IRBuilder<> B(SEL);
            Value* tDS = toDS(B, SEL->getTrueValue(),  tempOp1);
            Value* fDS = toDS(B, SEL->getFalseValue(), tempOp2);
            if (!tDS || !fDS) return;
            valueMap[SEL] = B.CreateSelect(SEL->getCondition(), tDS, fDS, SEL->getName());
            toDelete.push_back(SEL);
            return;
        }

        // --- fcmp: convert DS operands back to double for comparison ---
        if (auto* FC = dyn_cast<FCmpInst>(I)) {
            if (!FC->getOperand(0)->getType()->isDoubleTy()) return;
            bool lm = valueMap.count(FC->getOperand(0));
            bool rm = valueMap.count(FC->getOperand(1));
            if (!lm && !rm) return;
            IRBuilder<> B(FC);
            Value* lDbl = lm ? fromDS(B, valueMap[FC->getOperand(0)], tempOp1) : FC->getOperand(0);
            Value* rDbl = rm ? fromDS(B, valueMap[FC->getOperand(1)], tempOp2) : FC->getOperand(1);
            Value* newCmp = B.CreateFCmp(FC->getPredicate(), lDbl, rDbl, FC->getName());
            FC->replaceAllUsesWith(newCmp);
            toDelete.push_back(FC);
            return;
        }

        // --- sitofp / uitofp -> double: wrap result in DS ---
        if (auto* FP = dyn_cast<SIToFPInst>(I)) {
            if (!FP->getType()->isDoubleTy()) return;
            // Insert conversion after the sitofp (the sitofp stays)
            IRBuilder<> B(FP->getNextNode());
            B.CreateCall(convertToDSFunc, {tempResult, FP});
            valueMap[FP] = B.CreateLoad(dsType, tempResult);
            return; // do NOT add to toDelete — sitofp must stay for convertToDSFunc
        }
        if (auto* FP = dyn_cast<UIToFPInst>(I)) {
            if (!FP->getType()->isDoubleTy()) return;
            IRBuilder<> B(FP->getNextNode());
            B.CreateCall(convertToDSFunc, {tempResult, FP});
            valueMap[FP] = B.CreateLoad(dsType, tempResult);
            return;
        }

        // --- fptosi / fptoui: DS -> double -> int ---
        if (auto* FP = dyn_cast<FPToSIInst>(I)) {
            if (!FP->getOperand(0)->getType()->isDoubleTy()) return;
            if (!valueMap.count(FP->getOperand(0))) return;
            IRBuilder<> B(FP);
            Value* dbl    = fromDS(B, valueMap[FP->getOperand(0)], tempOp1);
            Value* intVal = B.CreateFPToSI(dbl, FP->getType());
            FP->replaceAllUsesWith(intVal);
            toDelete.push_back(FP);
            return;
        }
        if (auto* FP = dyn_cast<FPToUIInst>(I)) {
            if (!FP->getOperand(0)->getType()->isDoubleTy()) return;
            if (!valueMap.count(FP->getOperand(0))) return;
            IRBuilder<> B(FP);
            Value* dbl    = fromDS(B, valueMap[FP->getOperand(0)], tempOp1);
            Value* intVal = B.CreateFPToUI(dbl, FP->getType());
            FP->replaceAllUsesWith(intVal);
            toDelete.push_back(FP);
            return;
        }

        // --- fptrunc (double -> float): take high part ---
        if (auto* FP = dyn_cast<FPTruncInst>(I)) {
            if (!FP->getOperand(0)->getType()->isDoubleTy()) return;
            if (!FP->getType()->isFloatTy()) return;
            if (!valueMap.count(FP->getOperand(0))) return;
            IRBuilder<> B(FP);
            Value* hi = B.CreateExtractValue(valueMap[FP->getOperand(0)], {0}, FP->getName() + ".hi");
            FP->replaceAllUsesWith(hi);
            toDelete.push_back(FP);
            return;
        }

        // --- fpext (float -> double): DS with lo = 0 ---
        if (auto* FP = dyn_cast<FPExtInst>(I)) {
            if (!FP->getType()->isDoubleTy()) return;
            if (!FP->getOperand(0)->getType()->isFloatTy()) return;
            IRBuilder<> B(FP);
            Constant* z0  = ConstantFP::get(Type::getFloatTy(*ctx), 0.0f);
            Value*    dsV = B.CreateInsertValue(UndefValue::get(dsType), FP->getOperand(0), {0});
            dsV = B.CreateInsertValue(dsV, z0, {1});
            valueMap[FP] = dsV;
            toDelete.push_back(FP);
            return;
        }

        // --- PHI nodes ---
        if (auto* PHI = dyn_cast<PHINode>(I)) {
            if (!PHI->getType()->isDoubleTy()) return;
            IRBuilder<> B(PHI);
            PHINode* dsPhi = B.CreatePHI(dsType, PHI->getNumIncomingValues(),
                                          PHI->getName() + ".ds");
            valueMap[PHI] = dsPhi;
            phisToFix.push_back(dsPhi);
            toDelete.push_back(PHI);
            return;
        }

        // --- call instructions (intrinsics + external) ---
        if (auto* CI = dyn_cast<CallInst>(I)) {
            Function* fn = CI->getCalledFunction();
            if (!fn) return;

            switch (fn->getIntrinsicID()) {
            case Intrinsic::fmuladd: {
                if (!CI->getType()->isDoubleTy()) return;
                IRBuilder<> B(CI);
                Value* aDS = toDS(B, CI->getArgOperand(0), tempOp1);
                Value* bDS = toDS(B, CI->getArgOperand(1), tempOp2);
                Value* cDS = toDS(B, CI->getArgOperand(2), tempOp3);
                if (!aDS || !bDS || !cDS) return;
                B.CreateStore(aDS, tempOp1);
                B.CreateStore(bDS, tempOp2);
                B.CreateStore(cDS, tempOp3);
                B.CreateCall(fmaddFunc, {tempResult, tempOp1, tempOp2, tempOp3});
                valueMap[CI] = B.CreateLoad(dsType, tempResult);
                toDelete.push_back(CI);
                break;
            }
            case Intrinsic::sqrt:
                if (CI->getType()->isDoubleTy())
                    handleUnaryOp(CI, CI->getArgOperand(0), sqrtFunc);
                break;
            case Intrinsic::fabs:
                if (CI->getType()->isDoubleTy())
                    handleUnaryOp(CI, CI->getArgOperand(0), fabsFunc);
                break;
            case Intrinsic::memcpy: {
                // In typed-pointer IR, dst/src are bitcast to i8* — strip those casts.
                Value* dst = CI->getArgOperand(0)->stripPointerCasts();
                Value* src = CI->getArgOperand(1)->stripPointerCasts();
                if (valueMap.count(dst))
                    handleArrayInit(CI, dst, src, false);
                break;
            }
            case Intrinsic::memset: {
                Value* dst = CI->getArgOperand(0)->stripPointerCasts();
                Value* val = CI->getArgOperand(1);
                if (valueMap.count(dst) && isa<ConstantInt>(val) && cast<ConstantInt>(val)->isZero())
                    handleArrayInit(CI, dst, nullptr, true);
                break;
            }
            case Intrinsic::not_intrinsic:
                handleExternalCall(CI);
                break;
            default:
                break;
            }
            return;
        }

        // --- return: if returning double and value is in valueMap, convert back ---
        if (auto* RI = dyn_cast<ReturnInst>(I)) {
            Value* rv = RI->getReturnValue();
            if (!rv || !rv->getType()->isDoubleTy()) return;
            if (!valueMap.count(rv)) return;
            IRBuilder<> B(RI);
            Value* dbl = fromDS(B, valueMap[rv], tempOp1);
            RI->setOperand(0, dbl);
            return;
        }
    }

    // ----------------------------------------------------------------
    void fixupPHINodes() {
        for (PHINode* dsPhi : phisToFix) {
            PHINode* origPhi = nullptr;
            for (auto& kv : valueMap)
                if (kv.second == dsPhi) { origPhi = dyn_cast<PHINode>(kv.first); break; }
            if (!origPhi) continue;
            for (unsigned i = 0; i < origPhi->getNumIncomingValues(); i++) {
                Value*      inVal   = origPhi->getIncomingValue(i);
                BasicBlock* inBlock = origPhi->getIncomingBlock(i);
                if (valueMap.count(inVal)) {
                    dsPhi->addIncoming(valueMap[inVal], inBlock);
                } else if (auto* cfp = dyn_cast<ConstantFP>(inVal)) {
                    if (cfp->getType()->isDoubleTy()) {
                        // Convert constant at the end of the incoming block
                        IRBuilder<> B(inBlock->getTerminator());
                        B.CreateCall(convertToDSFunc, {tempResult, cfp});
                        dsPhi->addIncoming(B.CreateLoad(dsType, tempResult), inBlock);
                    }
                } else if (isa<UndefValue>(inVal) && inVal->getType()->isDoubleTy()) {
                    dsPhi->addIncoming(UndefValue::get(dsType), inBlock);
                }
            }
        }
    }

    bool isSafeToDelete(Instruction* I) {
        for (auto& use : I->uses()) {
            if (auto* ui = dyn_cast<Instruction>(use.getUser())) {
                if (!std::count(toDelete.begin(), toDelete.end(), ui)) return false;
            } else return false;
        }
        return true;
    }

    // ----------------------------------------------------------------
    PreservedAnalyses run(Module& M, ModuleAnalysisManager&) {
        ctx    = &M.getContext();
        mod    = &M;
        dsType = getDoubleSingleType(*ctx);
        declareExternalFunctions(M);

        for (auto& F : M) {
            if (F.empty()) continue;
            valueMap.clear();
            origTypeMap.clear();
            toDelete.clear();
            phisToFix.clear();
            createTemporaryStorage(F);

            // Snapshot original instructions before processing — prevents newly
            // inserted runtime calls from being re-processed by the iterator.
            std::vector<Instruction*> origInsts;
            for (auto& B : F)
                for (auto& I : B)
                    origInsts.push_back(&I);
            for (auto* I : origInsts)
                processInstruction(I);

            fixupPHINodes();

            for (auto it = toDelete.rbegin(); it != toDelete.rend(); ++it)
                if (isSafeToDelete(*it))
                    (*it)->eraseFromParent();
        }

        M.print(errs(), nullptr);
        return PreservedAnalyses::none();
    }
};

} // end anonymous namespace


extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "SkeletonPass", "v0.2",
        [](PassBuilder& PB) {
            PB.registerPipelineStartEPCallback(
                [](ModulePassManager& MPM, OptimizationLevel) {
                    MPM.addPass(SkeletonPass());
                });
        }
    };
}
