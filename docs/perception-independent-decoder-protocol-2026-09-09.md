# Independent local RGB and foreground decoders

Adaptive follow-up within the user's overnight authorization. The first D2 seed
shows local inputs help reconstruction and task FiLM trades a further RGB gain
against slightly lower foreground IoU. These reused-test observations motivated
this comparison; it is not a fresh confirmatory selection. Albedo stays excluded.

Question: does removing parameter sharing between typed RGB and foreground
decoders improve their output tradeoff when both receive identical local and
contextual evidence? Reference the completed D2 `early` and `conditioned` arms.
The new arm owns two complete early-input DenseDecoder branches, each initialized
to the shared unconditioned decoder's exact corresponding function. Keep only the
appropriate final output layer in each branch. Shared inputs/preprocessing/frozen
ViT remain identical. Two independent trunks nearly double decoder parameters;
this is a sharing/capacity/resource comparison, not a parameter-matched topology
test. One output uses one branch, both outputs use two passes.

Three paired seeds9107/9108/9109,4,000updates,64images/update as32COCO+32PushT, the
same recorded sampler stream, AdamW3e-4/wd1e-4, FP32/TF32off, frozen FP16final
features and FP32early patch inputs. Loss remains0.5COCO RGB MSE+0.5PushT RGB MSE+
1COCO valid-pixel foreground BCE. One optimizer and global norm1clipping over
all decoder parameters preserve the D2 clipping convention. Parameter sharing
is removed; global clipping can still couple gradient scales. Do not label the
disjoint trunks' gradient cosine as shared-feature interference. Log individual
losses and global gradient norm, with no new loss weights or normalization sweep.

All fits use the fixed4,000-update endpoint; full validation every100 is monitoring.
Evaluate the same512COCO/2506PushT test frames, full validation and512training-prefix
frames. Preserve per-image RGB/BCE/IoU/Dice, baseline outputs, first-six panels,
source/checkpoint identities, actual updates/frames, peak CUDA allocation and
per-output/both-output prepared-input timing. Primary endpoint is foreground IoU;
RGB errors and resource use remain separate outcomes. Report all paired seed
differences against D2early and conditioning without inventing a composite gate.

First test exact initial function matching and separate parameter ownership,
including an informative failure before implementation. Run one50-update CPU
development prefix through the full checkpoint/evaluation/HTML path, then commit
and seal source/data/configuration before formal GPU fits. D2's tested execution
helpers are reused unchanged; its archived source and active training files remain
untouched. The measured D2GPU fits finish within roughly8minutes, but retain a
conservative20minute cap per new fit (three fits<=60minutes), without extrapolating
CPU development timing into a GPU guarantee.

One GPU workload at a time. Wait for D2 completion and the queued localization_v2
development units, then run this three-fit comparison before the six formal
geometry-objective fits. Stop new training by06:00Berlin and report by07:00Berlin.
The complete program now contains36vision/geometry fits and15category probes.
No encoder updates, future-pixel shortcuts, new datasets or control claims.
