/**
 * app.js -- Water Body Segmentation web UI
 *
 * No frameworks, no build step. Talks to the existing /predict and
 * /health endpoints exactly as they already are -- the JSON response
 * shape here (width, height, num_tiles, water_coverage_pct, threshold,
 * device, timings.{load,tile,inference,stitch,total}, mask_png_base64)
 * matches serve.py's PredictResponse model precisely; if that model
 * ever changes, this file needs to change with it, not the other way
 * around.
 */

(() => {
  "use strict";

  // ---- DOM references -----------------------------------------------
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");
  const previewSection = document.getElementById("preview-section");
  const previewImg = document.getElementById("preview-img");
  const previewFallback = document.getElementById("preview-fallback");
  const runBtn = document.getElementById("run-btn");
  const resetBtn = document.getElementById("reset-btn");

  const loadingSection = document.getElementById("loading-section");
  const errorSection = document.getElementById("error-section");
  const errorMessage = document.getElementById("error-message");
  const resultsSection = document.getElementById("results-section");

  const panelOriginal = document.getElementById("panel-original");
  const panelMask = document.getElementById("panel-mask");
  const overlayBase = document.getElementById("overlay-base");
  const overlayTint = document.getElementById("overlay-tint");
  const opacitySlider = document.getElementById("opacity-slider");
  const opacityValue = document.getElementById("opacity-value");

  const statCoverage = document.getElementById("stat-coverage");
  const statTime = document.getElementById("stat-time");
  const statWidth = document.getElementById("stat-width");
  const statHeight = document.getElementById("stat-height");
  const statTiles = document.getElementById("stat-tiles");
  const statDevice = document.getElementById("stat-device");

  const downloadMaskBtn = document.getElementById("download-mask-btn");
  const downloadOverlayBtn = document.getElementById("download-overlay-btn");

  const modelInfoBar = document.getElementById("model-info-bar");

  // ---- State -----------------------------------------------------------
  let selectedFile = null;
  let originalObjectUrl = null;
  let maskObjectUrl = null;
  let lastFilenameBase = "result";

  // ============================================================
  // Model info (fetched once on load, from the existing /health)
  // ============================================================
  async function loadModelInfo() {
    try {
      const res = await fetch("/health");
      if (!res.ok) return; // non-fatal -- the info bar just stays hidden
      const info = await res.json();
      modelInfoBar.innerHTML = "";
      const fields = [
        ["Model", info.model_name],
        ["Device", info.device],
        ["Tile size", info.tile_size],
        ["Overlap", info.overlap],
        ["Patch size", info.patch_size],
        ["Threshold", info.threshold],
        ["Version", info.model_version],
      ];
      for (const [label, value] of fields) {
        const chip = document.createElement("span");
        chip.className = "info-chip";
        chip.innerHTML = `<span class="info-chip-label">${label}</span><span class="info-chip-value">${value}</span>`;
        modelInfoBar.appendChild(chip);
      }
      modelInfoBar.hidden = false;
    } catch (e) {
      // Server unreachable at load time -- not fatal, the rest of the
      // page still works and /predict will surface its own error later.
    }
  }

  // ============================================================
  // Upload: drag & drop + browse
  // ============================================================
  function isSupportedImage(file) {
    // Matches what the backend accepts (content-type starting with
    // "image/") plus extension-based checks for formats browsers
    // sometimes mis-type (e.g. .tif/.tiff, GeoTIFF).
    if (file.type && file.type.startsWith("image/")) return true;
    return /\.(jpe?g|png|tiff?|bmp|webp)$/i.test(file.name);
  }

  function handleFileSelected(file) {
    if (!file) return;
    if (!isSupportedImage(file)) {
      showError(`"${file.name}" doesn't look like a supported image file (JPG, PNG, TIFF, BMP, WEBP).`);
      return;
    }

    hideError();
    selectedFile = file;
    lastFilenameBase = file.name.replace(/\.[^.]+$/, "") || "result";

    if (originalObjectUrl) URL.revokeObjectURL(originalObjectUrl);
    originalObjectUrl = URL.createObjectURL(file);

    // Browsers can't natively render every format this service accepts
    // for inference -- confirmed directly: TIFF/GeoTIFF has no <img>
    // support in Chrome, Firefox, or Edge at all. At this pre-inference
    // stage there's no server-converted version to fall back to yet (no
    // round trip has happened), so this degrades to a text confirmation
    // instead of showing a broken-image icon -- the actual visual
    // preview appears after inference, using the server-side conversion
    // in renderResults() below.
    previewImg.hidden = false;
    previewFallback.hidden = true;
    previewImg.onerror = () => {
      previewImg.hidden = true;
      previewFallback.hidden = false;
      previewFallback.textContent = `Selected: ${file.name} (preview unavailable for this format, inference will still run)`;
    };
    previewImg.src = originalObjectUrl;

    previewSection.hidden = false;
    resultsSection.hidden = true;
    runBtn.disabled = false;
  }

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-active");
  });
  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-active");
  });
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-active");
    const file = e.dataTransfer.files[0];
    handleFileSelected(file);
  });
  dropZone.addEventListener("click", () => fileInput.click());
  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });
  fileInput.addEventListener("change", () => handleFileSelected(fileInput.files[0]));

  // ============================================================
  // Run segmentation
  // ============================================================
  runBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    hideError();
    resultsSection.hidden = true;
    loadingSection.hidden = false;
    runBtn.disabled = true;

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await fetch("/predict", { method: "POST", body: formData });

      if (!res.ok) {
        let detail = `Server returned ${res.status}`;
        try {
          const body = await res.json();
          if (body.detail) detail = body.detail;
        } catch (_) {
          /* response wasn't JSON -- keep the generic message */
        }
        throw new Error(detail);
      }

      const data = await res.json();
      await renderResults(data);
    } catch (err) {
      if (err instanceof TypeError) {
        // fetch() throws a bare TypeError for network-level failures
        // (server unreachable, connection refused) -- distinguish this
        // from a real HTTP error response, which has already been
        // turned into a proper Error with a useful message above.
        showError("Couldn't reach the server. It may be starting up or unavailable -- try again in a moment.");
      } else {
        showError(err.message || "Something went wrong during inference.");
      }
    } finally {
      loadingSection.hidden = true;
      runBtn.disabled = false;
    }
  });

  resetBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    previewSection.hidden = true;
    resultsSection.hidden = true;
    hideError();
    runBtn.disabled = true;
  });

  // ============================================================
  // Render results: three panels, stats, overlay, downloads
  // ============================================================
  async function renderResults(data) {
    if (maskObjectUrl) URL.revokeObjectURL(maskObjectUrl);

    // Uses the server-converted PNG (data.original_png_base64), not the
    // raw uploaded blob (originalObjectUrl) -- the raw file is whatever
    // format was uploaded, which the browser may not support displaying
    // at all (confirmed: GeoTIFF has no native <img> support in any
    // major browser, even though the server reads it fine for
    // inference). The server already decoded it successfully to run
    // inference, so re-encoding that same decode as PNG here guarantees
    // something every browser can actually show.
    const originalDataUrl = `data:image/png;base64,${data.original_png_base64}`;
    const maskDataUrl = `data:image/png;base64,${data.mask_png_base64}`;

    panelOriginal.src = originalDataUrl;
    panelMask.src = maskDataUrl;
    overlayBase.src = originalDataUrl;

    // The tinted overlay layer is generated on a canvas (recoloring the
    // binary black/white mask into a translucent cyan tint with
    // transparent "no water" pixels) rather than just stacking the raw
    // mask image -- a plain grayscale mask on top would look like a
    // literal black rectangle at any opacity above ~0, not a highlight.
    const tintedUrl = await buildTintedOverlay(maskDataUrl);
    overlayTint.src = tintedUrl;
    overlayTint.style.opacity = opacitySlider.value / 100;

    statCoverage.textContent = `${data.water_coverage_pct.toFixed(2)}%`;
    statTime.textContent = `${data.timings.total.toFixed(2)}s`;
    statWidth.textContent = `${data.width}px`;
    statHeight.textContent = `${data.height}px`;
    statTiles.textContent = data.num_tiles;
    statDevice.textContent = data.device.toUpperCase();

    resultsSection.hidden = false;
  }

  opacitySlider.addEventListener("input", () => {
    const pct = opacitySlider.value;
    opacityValue.textContent = `${pct}%`;
    overlayTint.style.opacity = pct / 100;
  });

  /**
   * Recolors a binary black/white mask PNG into a translucent cyan
   * overlay: white (water) pixels become semi-opaque cyan, black
   * pixels become fully transparent. Returns a data URL.
   */
  function buildTintedOverlay(maskDataUrl) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const px = imageData.data;
        for (let i = 0; i < px.length; i += 4) {
          const isWater = px[i] > 127; // mask is binary: 0 or 255 per channel
          px[i] = 56; // R
          px[i + 1] = 224; // G
          px[i + 2] = 255; // B
          px[i + 3] = isWater ? 190 : 0; // alpha
        }
        ctx.putImageData(imageData, 0, 0);
        resolve(canvas.toDataURL("image/png"));
      };
      img.onerror = reject;
      img.src = maskDataUrl;
    });
  }

  // ============================================================
  // Downloads -- both fully client-side, from data already in the
  // browser (no extra request to the server needed).
  // ============================================================
  function triggerDownload(dataUrl, filename) {
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  downloadMaskBtn.addEventListener("click", () => {
    triggerDownload(panelMask.src, `${lastFilenameBase}_mask.png`);
  });

  downloadOverlayBtn.addEventListener("click", () => {
    // Flattens the base image + current tint opacity onto one canvas --
    // "preserve the exact displayed result" means baking in whatever
    // opacity the slider is currently set to, not always 100%.
    const canvas = document.createElement("canvas");
    canvas.width = overlayBase.naturalWidth;
    canvas.height = overlayBase.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(overlayBase, 0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = opacitySlider.value / 100;
    ctx.drawImage(overlayTint, 0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = 1;
    triggerDownload(canvas.toDataURL("image/png"), `${lastFilenameBase}_overlay.png`);
  });

  // ============================================================
  // Error display
  // ============================================================
  function showError(message) {
    errorMessage.textContent = message;
    errorSection.hidden = false;
  }
  function hideError() {
    errorSection.hidden = true;
  }

  loadModelInfo();
})();
