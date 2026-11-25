# Changelog

## v1.0.0
- Initial public release of **DIP Studio** - Digital Image Processing Studio.
- **112 non-destructive operations across 13 categories**, including **AI Studio (ONNX)** with automatic GPU hardware acceleration (`CUDA`, `CoreML`, `DirectML`, `ROCm`, `OpenVINO`) and safe CPU fallback:
  - **19 AI Studio Tools:** Super-Resolution (ESPCN 3x & 4x-UltraSharp), Background Removal & Die-Cut Sticker Cutout (RMBG-1.4, MODNet, U2-Net-p), Color Splash (Subject Pop), Portrait Bokeh, 3D Depth Map, 3D Depth Focus, 3D Depth Fog & 3D Depth Relighting (Depth Anything V2), Object Detection & Object Privacy Redactor (YOLOv8n), Anime / Illustration (AnimeGANv2), Face & Landmark Detection, Face Privacy, Smart Portrait Auto-Crop (Close-up Face default) & Portrait Retouch (YuNet), Neural Style Transfer, Low-Light Night Enhance, and Smart Auto-Enhance.
  - Full classic CV & Chapters 3-7 coverage: Multi-Level/Iterative `T*` & Niblack/Sauvola thresholds, Morphological Hole Filling, Convex Hull, Hit-or-Miss, Zhang-Suen Thinning/Pruning, Point & Line masks, HSV Color Slicing, YCbCr Skin Detector, Color Edge Vector Gradients, Region Growing & Quadtree Split-Merge, Scale-Space Blobs (LoG/DoG/DoH), LBP/Gabor/GLCM Texture, HOG, and SIFT/ORB Local Descriptors.
  - Complete Light/Dark `QPalette` support across dropdown popups, dialogs, tooltips, and controls; minimal non-clipping sliders; responsive reflowing Histogram panel; clean process shutdown on window close.
- Smart prefix-cached pipeline evaluation with 60 FPS fast-path previews, cancellable warm subprocess workers, and descending step stack (`N..1`) preserving RGBA alpha transparency across downstream operations.
- Relative-scale synchronized Side-by-Side, Split Curtain, Difference Heatmap, and Result Only comparison views, interactive Crop & Aspect Frame overlay, and **Transparent** border fill default for geometric/affine warps.
- Live 256-bin Histogram, CDF, Image Quality Metrics (PSNR, SSIM, MSE, MAE), Transfer Curve visualizer, 3×3/5×5 Convolution Editor, Structuring Element preview, and 2×3 Affine Matrix inspector.
- OS system theme auto-detection on first launch, 16 bundled sample images, one-click workflow presets, `.dips.json` preset import/export, batch folder processing, PEP 621 versioning (`pyproject.toml`), and native builds for Windows, macOS, and Linux.
