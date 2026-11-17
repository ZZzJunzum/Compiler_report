; ModuleID = '__compute_module_compare_convert_fusion_kernel_module'
source_filename = "__compute_module_compare_convert_fusion_kernel_module"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-win32"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable
define noalias noundef ptr @compare_convert_fusion(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !4
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %8 = load ptr, ptr %7, align 8, !invariant.load !3, !dereferenceable !4
  %9 = getelementptr inbounds nuw i8, ptr %3, i64 48
  %10 = load ptr, ptr %9, align 8, !invariant.load !3, !dereferenceable !5
  %11 = getelementptr inbounds nuw i8, ptr %3, i64 64
  %12 = load ptr, ptr %11, align 8, !invariant.load !3, !dereferenceable !4
  %13 = getelementptr inbounds nuw i8, ptr %3, i64 80
  %14 = load ptr, ptr %13, align 8, !invariant.load !3, !dereferenceable !5
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %16 = load ptr, ptr %15, align 8
  %17 = load i64, ptr %16, align 4, !invariant.load !3
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !9)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !11)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !13)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !15)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !17)
  %18 = icmp ult i64 %17, 4
  br i1 %18, label %19, label %compare_convert_fusion_wrapped.exit

19:                                               ; preds = %1
  %20 = shl nuw nsw i64 %17, 18
  br label %vector.ph

vector.ph:                                        ; preds = %19, %middle.block
  %21 = phi i64 [ 0, %19 ], [ %79, %middle.block ]
  %22 = shl nuw nsw i64 %21, 8
  %23 = add nuw nsw i64 %22, %20
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next.1, %vector.body ]
  %24 = add nuw nsw i64 %index, %23
  %25 = getelementptr inbounds nuw float, ptr %10, i64 %24
  %26 = getelementptr inbounds nuw i8, ptr %25, i64 32
  %wide.load = load <8 x float>, ptr %25, align 4, !invariant.load !3, !alias.scope !13, !noalias !19
  %wide.load5 = load <8 x float>, ptr %26, align 4, !invariant.load !3, !alias.scope !13, !noalias !19
  %27 = getelementptr inbounds nuw float, ptr %12, i64 %index
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 32
  %wide.load6 = load <8 x float>, ptr %27, align 4, !invariant.load !3, !alias.scope !15, !noalias !20
  %wide.load7 = load <8 x float>, ptr %28, align 4, !invariant.load !3, !alias.scope !15, !noalias !20
  %29 = fmul <8 x float> %wide.load, %wide.load6
  %30 = fmul <8 x float> %wide.load5, %wide.load7
  %31 = getelementptr inbounds nuw float, ptr %8, i64 %index
  %32 = getelementptr inbounds nuw i8, ptr %31, i64 32
  %wide.load8 = load <8 x float>, ptr %31, align 4, !invariant.load !3, !alias.scope !11, !noalias !21
  %wide.load9 = load <8 x float>, ptr %32, align 4, !invariant.load !3, !alias.scope !11, !noalias !21
  %33 = fadd <8 x float> %29, %wide.load8
  %34 = fadd <8 x float> %30, %wide.load9
  %35 = tail call <8 x float> @llvm.maximum.v8f32(<8 x float> %33, <8 x float> zeroinitializer)
  %36 = tail call <8 x float> @llvm.maximum.v8f32(<8 x float> %34, <8 x float> zeroinitializer)
  %37 = getelementptr inbounds nuw float, ptr %6, i64 %index
  %38 = getelementptr inbounds nuw i8, ptr %37, i64 32
  %wide.load10 = load <8 x float>, ptr %37, align 4, !invariant.load !3, !alias.scope !9, !noalias !22
  %wide.load11 = load <8 x float>, ptr %38, align 4, !invariant.load !3, !alias.scope !9, !noalias !22
  %39 = fmul <8 x float> %wide.load10, %35
  %40 = fmul <8 x float> %wide.load11, %36
  %41 = getelementptr inbounds nuw float, ptr %4, i64 %index
  %42 = getelementptr inbounds nuw i8, ptr %41, i64 32
  %wide.load12 = load <8 x float>, ptr %41, align 4, !invariant.load !3, !alias.scope !6, !noalias !23
  %wide.load13 = load <8 x float>, ptr %42, align 4, !invariant.load !3, !alias.scope !6, !noalias !23
  %43 = fadd <8 x float> %wide.load12, %39
  %44 = fadd <8 x float> %wide.load13, %40
  %45 = fcmp ogt <8 x float> %43, zeroinitializer
  %46 = fcmp ogt <8 x float> %44, zeroinitializer
  %47 = zext <8 x i1> %45 to <8 x i32>
  %48 = zext <8 x i1> %46 to <8 x i32>
  %49 = getelementptr inbounds nuw i32, ptr %14, i64 %24
  %50 = getelementptr inbounds nuw i8, ptr %49, i64 32
  store <8 x i32> %47, ptr %49, align 4, !alias.scope !17, !noalias !24
  store <8 x i32> %48, ptr %50, align 4, !alias.scope !17, !noalias !24
  %index.next = or disjoint i64 %index, 16
  %51 = add nuw nsw i64 %index.next, %23
  %52 = getelementptr inbounds nuw float, ptr %10, i64 %51
  %53 = getelementptr inbounds nuw i8, ptr %52, i64 32
  %wide.load.1 = load <8 x float>, ptr %52, align 4, !invariant.load !3, !alias.scope !13, !noalias !19
  %wide.load5.1 = load <8 x float>, ptr %53, align 4, !invariant.load !3, !alias.scope !13, !noalias !19
  %54 = getelementptr inbounds nuw float, ptr %12, i64 %index.next
  %55 = getelementptr inbounds nuw i8, ptr %54, i64 32
  %wide.load6.1 = load <8 x float>, ptr %54, align 4, !invariant.load !3, !alias.scope !15, !noalias !20
  %wide.load7.1 = load <8 x float>, ptr %55, align 4, !invariant.load !3, !alias.scope !15, !noalias !20
  %56 = fmul <8 x float> %wide.load.1, %wide.load6.1
  %57 = fmul <8 x float> %wide.load5.1, %wide.load7.1
  %58 = getelementptr inbounds nuw float, ptr %8, i64 %index.next
  %59 = getelementptr inbounds nuw i8, ptr %58, i64 32
  %wide.load8.1 = load <8 x float>, ptr %58, align 4, !invariant.load !3, !alias.scope !11, !noalias !21
  %wide.load9.1 = load <8 x float>, ptr %59, align 4, !invariant.load !3, !alias.scope !11, !noalias !21
  %60 = fadd <8 x float> %56, %wide.load8.1
  %61 = fadd <8 x float> %57, %wide.load9.1
  %62 = tail call <8 x float> @llvm.maximum.v8f32(<8 x float> %60, <8 x float> zeroinitializer)
  %63 = tail call <8 x float> @llvm.maximum.v8f32(<8 x float> %61, <8 x float> zeroinitializer)
  %64 = getelementptr inbounds nuw float, ptr %6, i64 %index.next
  %65 = getelementptr inbounds nuw i8, ptr %64, i64 32
  %wide.load10.1 = load <8 x float>, ptr %64, align 4, !invariant.load !3, !alias.scope !9, !noalias !22
  %wide.load11.1 = load <8 x float>, ptr %65, align 4, !invariant.load !3, !alias.scope !9, !noalias !22
  %66 = fmul <8 x float> %wide.load10.1, %62
  %67 = fmul <8 x float> %wide.load11.1, %63
  %68 = getelementptr inbounds nuw float, ptr %4, i64 %index.next
  %69 = getelementptr inbounds nuw i8, ptr %68, i64 32
  %wide.load12.1 = load <8 x float>, ptr %68, align 4, !invariant.load !3, !alias.scope !6, !noalias !23
  %wide.load13.1 = load <8 x float>, ptr %69, align 4, !invariant.load !3, !alias.scope !6, !noalias !23
  %70 = fadd <8 x float> %wide.load12.1, %66
  %71 = fadd <8 x float> %wide.load13.1, %67
  %72 = fcmp ogt <8 x float> %70, zeroinitializer
  %73 = fcmp ogt <8 x float> %71, zeroinitializer
  %74 = zext <8 x i1> %72 to <8 x i32>
  %75 = zext <8 x i1> %73 to <8 x i32>
  %76 = getelementptr inbounds nuw i32, ptr %14, i64 %51
  %77 = getelementptr inbounds nuw i8, ptr %76, i64 32
  store <8 x i32> %74, ptr %76, align 4, !alias.scope !17, !noalias !24
  store <8 x i32> %75, ptr %77, align 4, !alias.scope !17, !noalias !24
  %index.next.1 = add nuw nsw i64 %index, 32
  %78 = icmp eq i64 %index.next.1, 256
  br i1 %78, label %middle.block, label %vector.body, !llvm.loop !25

middle.block:                                     ; preds = %vector.body
  %79 = add nuw nsw i64 %21, 1
  %exitcond3.not = icmp eq i64 %79, 1024
  br i1 %exitcond3.not, label %compare_convert_fusion_wrapped.exit, label %vector.ph, !llvm.loop !28

compare_convert_fusion_wrapped.exit:              ; preds = %middle.block, %1
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.maximum.v8f32(<8 x float>, <8 x float>) #2

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 0}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 1024}
!5 = !{i64 4194304}
!6 = !{!7}
!7 = distinct !{!7, !8, !"compare_convert_fusion_wrapped: argument 0"}
!8 = distinct !{!8, !"compare_convert_fusion_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"compare_convert_fusion_wrapped: argument 1"}
!11 = !{!12}
!12 = distinct !{!12, !8, !"compare_convert_fusion_wrapped: argument 2"}
!13 = !{!14}
!14 = distinct !{!14, !8, !"compare_convert_fusion_wrapped: argument 3"}
!15 = !{!16}
!16 = distinct !{!16, !8, !"compare_convert_fusion_wrapped: argument 4"}
!17 = !{!18}
!18 = distinct !{!18, !8, !"compare_convert_fusion_wrapped: argument 5"}
!19 = !{!7, !10, !12, !16, !18}
!20 = !{!7, !10, !12, !14, !18}
!21 = !{!7, !10, !14, !16, !18}
!22 = !{!7, !12, !14, !16, !18}
!23 = !{!10, !12, !14, !16, !18}
!24 = !{!7, !10, !12, !14, !16}
!25 = distinct !{!25, !26, !27}
!26 = !{!"llvm.loop.isvectorized", i32 1}
!27 = !{!"llvm.loop.unroll.runtime.disable"}
!28 = distinct !{!28, !29}
!29 = !{!"llvm.loop.unroll.disable"}
