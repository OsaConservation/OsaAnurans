// Share observers so removed, offscreen cards do not retain recordings and callbacks.
const pending = new Map();
let intersection;
let removals;

export function whenVisible(node, callback) {
    if (!intersection) {
        intersection = new IntersectionObserver((entries) => {
            for (const entry of entries) {
                if (!entry.isIntersecting) continue;
                const run = pending.get(entry.target);
                forget(entry.target);
                run?.();
            }
        });
        removals = new MutationObserver(() => {
            for (const target of pending.keys()) {
                if (!target.isConnected) forget(target);
            }
        });
    }

    pending.set(node, callback);
    intersection.observe(node);
    removals.observe(document.documentElement, { childList: true, subtree: true });
}

function forget(node) {
    pending.delete(node);
    intersection.unobserve(node);
    if (pending.size === 0) removals.disconnect();
}
