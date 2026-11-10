// Enhance only annotation spectrograms, including images generated after loading.
export function installSpectrogramPreview(workspace) {
    const dialog = document.createElement("dialog");
    dialog.className = "spectrogram-preview";
    dialog.setAttribute("aria-labelledby", "spectrogram-preview-title");
    const heading = document.createElement("h2");
    heading.id = "spectrogram-preview-title";
    const close = document.createElement("button");
    close.type = "button";
    close.className = "spectrogram-preview-close";
    close.textContent = "×";
    close.setAttribute("aria-label", "Close spectrogram preview");
    const image = document.createElement("img");
    const error = document.createElement("p");
    error.setAttribute("role", "status");
    error.hidden = true;
    error.textContent = "This spectrogram could not be loaded. Close the preview and retry the image.";
    image.addEventListener("error", () => { image.hidden = true; error.hidden = false; });
    dialog.append(heading, close, image, error);
    document.body.append(dialog);

    let source;
    close.addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (event) => {
        if (event.target !== dialog) return;
        const bounds = dialog.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right ||
            event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
    });
    dialog.addEventListener("close", () => {
        document.body.classList.remove("spectrogram-preview-open");
        image.removeAttribute("src");
        if (source?.isConnected && !workspace.hidden) source.focus({ preventScroll: true });
        source = null;
    });

    function open(target) {
        if (workspace.hidden || dialog.open || !target.complete || !target.naturalWidth) return;
        source = target;
        const candidate = target.closest(".results-list li");
        const species = target.closest(".frog-card")?.querySelector("h3")?.textContent;
        const label = candidate?.querySelector("h3")?.textContent || species || target.alt;
        heading.textContent = label || "Spectrogram preview";
        image.alt = target.alt || "Enlarged spectrogram";
        error.hidden = true;
        image.hidden = false;
        image.src = target.currentSrc || target.src;
        dialog.showModal();
        document.body.classList.add("spectrogram-preview-open");
        close.focus();
    }

    function enhance(target) {
        if (!target.matches(".audio-clip img") || target.dataset.previewEnabled) return;
        target.dataset.previewEnabled = "true";
        target.tabIndex = 0;
        target.setAttribute("role", "button");
        target.setAttribute("aria-haspopup", "dialog");
        target.setAttribute("aria-label", `Enlarge ${target.alt || "spectrogram"}`);
        target.title = "Click to enlarge spectrogram";
        target.addEventListener("click", () => open(target));
        target.addEventListener("keydown", (event) => {
            if (event.key !== "Enter" && event.key !== " ") return;
            event.preventDefault();
            open(target);
        });
    }

    workspace.querySelectorAll(".audio-clip img").forEach(enhance);
    const observer = new MutationObserver((records) => {
        for (const record of records) for (const node of record.addedNodes) {
            if (node.nodeType !== Node.ELEMENT_NODE) continue;
            if (node.matches("img")) enhance(node);
            node.querySelectorAll("img").forEach(enhance);
        }
    });
    observer.observe(workspace, { childList: true, subtree: true });
    return { close() { if (dialog.open) dialog.close(); } };
}
