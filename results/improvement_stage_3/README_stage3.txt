Stage 3 improvement candidates:
1) se_unet focal_dice tiles_per_image=4
2) se_unet focal_dice tiles_per_image=6
3) se_unet focal_tversky tiles_per_image=4
4) cbam_unet focal_dice tiles_per_image=4
5) dual_encoder_unet focal_dice tiles_per_image=4
Selection metric: mean GroupKFold validation Dice.
Held-out test and tiled full-image evaluation are run only after selecting best validation model.
