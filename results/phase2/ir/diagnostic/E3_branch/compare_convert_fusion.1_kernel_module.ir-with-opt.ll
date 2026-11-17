; ModuleID = '__compute_module_compare_convert_fusion.1_kernel_module'
source_filename = "__compute_module_compare_convert_fusion.1_kernel_module"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-win32"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable
define noalias noundef ptr @compare_convert_fusion.1(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !5
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %8 = load ptr, ptr %7, align 8, !invariant.load !3, !dereferenceable !4
  %9 = getelementptr inbounds nuw i8, ptr %3, i64 48
  %10 = load ptr, ptr %9, align 8, !invariant.load !3, !dereferenceable !5
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %12 = load ptr, ptr %11, align 8
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !9)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !11)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !13)
  %14 = icmp ult i64 %13, 4
  br i1 %14, label %15, label %compare_convert_fusion.1_wrapped.exit

15:                                               ; preds = %1
  %16 = shl nuw nsw i64 %13, 18
  br label %vector.ph

vector.ph:                                        ; preds = %15, %middle.block
  %17 = phi i64 [ 0, %15 ], [ %55, %middle.block ]
  %18 = shl nuw nsw i64 %17, 8
  %19 = add nuw nsw i64 %18, %16
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next.1, %vector.body ]
  %20 = add nuw nsw i64 %index, %19
  %21 = getelementptr inbounds nuw float, ptr %6, i64 %20
  %22 = getelementptr inbounds nuw i8, ptr %21, i64 32
  %wide.load = load <8 x float>, ptr %21, align 4, !invariant.load !3, !alias.scope !9, !noalias !15
  %wide.load5 = load <8 x float>, ptr %22, align 4, !invariant.load !3, !alias.scope !9, !noalias !15
  %23 = getelementptr inbounds nuw float, ptr %8, i64 %index
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 32
  %wide.load6 = load <8 x float>, ptr %23, align 4, !invariant.load !3, !alias.scope !11, !noalias !16
  %wide.load7 = load <8 x float>, ptr %24, align 4, !invariant.load !3, !alias.scope !11, !noalias !16
  %25 = fmul <8 x float> %wide.load, %wide.load6
  %26 = fmul <8 x float> %wide.load5, %wide.load7
  %27 = getelementptr inbounds nuw float, ptr %4, i64 %index
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 32
  %wide.load8 = load <8 x float>, ptr %27, align 4, !invariant.load !3, !alias.scope !6, !noalias !17
  %wide.load9 = load <8 x float>, ptr %28, align 4, !invariant.load !3, !alias.scope !6, !noalias !17
  %29 = fadd <8 x float> %25, %wide.load8
  %30 = fadd <8 x float> %26, %wide.load9
  %31 = fcmp ogt <8 x float> %29, zeroinitializer
  %32 = fcmp ogt <8 x float> %30, zeroinitializer
  %33 = zext <8 x i1> %31 to <8 x i32>
  %34 = zext <8 x i1> %32 to <8 x i32>
  %35 = getelementptr inbounds nuw i32, ptr %10, i64 %20
  %36 = getelementptr inbounds nuw i8, ptr %35, i64 32
  store <8 x i32> %33, ptr %35, align 4, !alias.scope !13, !noalias !18
  store <8 x i32> %34, ptr %36, align 4, !alias.scope !13, !noalias !18
  %index.next = or disjoint i64 %index, 16
  %37 = add nuw nsw i64 %index.next, %19
  %38 = getelementptr inbounds nuw float, ptr %6, i64 %37
  %39 = getelementptr inbounds nuw i8, ptr %38, i64 32
  %wide.load.1 = load <8 x float>, ptr %38, align 4, !invariant.load !3, !alias.scope !9, !noalias !15
  %wide.load5.1 = load <8 x float>, ptr %39, align 4, !invariant.load !3, !alias.scope !9, !noalias !15
  %40 = getelementptr inbounds nuw float, ptr %8, i64 %index.next
  %41 = getelementptr inbounds nuw i8, ptr %40, i64 32
  %wide.load6.1 = load <8 x float>, ptr %40, align 4, !invariant.load !3, !alias.scope !11, !noalias !16
  %wide.load7.1 = load <8 x float>, ptr %41, align 4, !invariant.load !3, !alias.scope !11, !noalias !16
  %42 = fmul <8 x float> %wide.load.1, %wide.load6.1
  %43 = fmul <8 x float> %wide.load5.1, %wide.load7.1
  %44 = getelementptr inbounds nuw float, ptr %4, i64 %index.next
  %45 = getelementptr inbounds nuw i8, ptr %44, i64 32
  %wide.load8.1 = load <8 x float>, ptr %44, align 4, !invariant.load !3, !alias.scope !6, !noalias !17
  %wide.load9.1 = load <8 x float>, ptr %45, align 4, !invariant.load !3, !alias.scope !6, !noalias !17
  %46 = fadd <8 x float> %42, %wide.load8.1
  %47 = fadd <8 x float> %43, %wide.load9.1
  %48 = fcmp ogt <8 x float> %46, zeroinitializer
  %49 = fcmp ogt <8 x float> %47, zeroinitializer
  %50 = zext <8 x i1> %48 to <8 x i32>
  %51 = zext <8 x i1> %49 to <8 x i32>
  %52 = getelementptr inbounds nuw i32, ptr %10, i64 %37
  %53 = getelementptr inbounds nuw i8, ptr %52, i64 32
  store <8 x i32> %50, ptr %52, align 4, !alias.scope !13, !noalias !18
  store <8 x i32> %51, ptr %53, align 4, !alias.scope !13, !noalias !18
  %index.next.1 = add nuw nsw i64 %index, 32
  %54 = icmp eq i64 %index.next.1, 256
  br i1 %54, label %middle.block, label %vector.body, !llvm.loop !19

middle.block:                                     ; preds = %vector.body
  %55 = add nuw nsw i64 %17, 1
  %exitcond3.not = icmp eq i64 %55, 1024
  br i1 %exitcond3.not, label %compare_convert_fusion.1_wrapped.exit, label %vector.ph, !llvm.loop !22

compare_convert_fusion.1_wrapped.exit:            ; preds = %middle.block, %1
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 1}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 1024}
!5 = !{i64 4194304}
!6 = !{!7}
!7 = distinct !{!7, !8, !"compare_convert_fusion.1_wrapped: argument 0"}
!8 = distinct !{!8, !"compare_convert_fusion.1_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"compare_convert_fusion.1_wrapped: argument 1"}
!11 = !{!12}
!12 = distinct !{!12, !8, !"compare_convert_fusion.1_wrapped: argument 2"}
!13 = !{!14}
!14 = distinct !{!14, !8, !"compare_convert_fusion.1_wrapped: argument 3"}
!15 = !{!7, !12, !14}
!16 = !{!7, !10, !14}
!17 = !{!10, !12, !14}
!18 = !{!7, !10, !12}
!19 = distinct !{!19, !20, !21}
!20 = !{!"llvm.loop.isvectorized", i32 1}
!21 = !{!"llvm.loop.unroll.runtime.disable"}
!22 = distinct !{!22, !23}
!23 = !{!"llvm.loop.unroll.disable"}
