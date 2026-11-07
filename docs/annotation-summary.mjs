import { element } from "./annotation-dom.mjs";

export function sortSummary(rows, key = "total", descending = key === "total") {
    return [...rows].sort((a, b) => {
        const comparison = key === "scientificName"
            ? a.scientificName.localeCompare(b.scientificName)
            : a.counts[key] - b.counts[key];
        return (descending ? -comparison : comparison) || a.scientificName.localeCompare(b.scientificName);
    });
}

export function createAnnotationSummary({ provider, available, select, locked }) {
    const section = document.querySelector("#annotation-summary");
    const status = document.querySelector("#summary-status");
    const refreshButton = document.querySelector("#refresh-summary");
    const table = document.querySelector("#species-summary");
    let rows = [], key = "total", descending = true, refreshing = false, refreshAgain = false;
    section.hidden = false;

    function render() {
        table.querySelectorAll("thead th").forEach((th) => {
            const active = th.querySelector("button").dataset.sort === key;
            if (active) th.setAttribute("aria-sort", descending ? "descending" : "ascending");
            else th.removeAttribute("aria-sort");
        });
        const body = table.querySelector("tbody");
        body.replaceChildren();
        for (const row of sortSummary(rows, key, descending)) {
            const tr = element("tr"), name = element("th");
            name.scope = "row";
            if (available(row.id)) {
                const button = element("button", row.scientificName, "summary-species");
                button.type = "button";
                button.disabled = locked();
                button.addEventListener("click", () => { if (!locked()) select(row.id); });
                name.append(button);
            } else {
                name.append(document.createTextNode(row.scientificName), element("small", "No review clips available"));
            }
            tr.append(name);
            for (const field of ["positive", "negative", "unsure", "total"]) {
                tr.append(element("td", String(row.counts[field])));
            }
            body.append(tr);
        }
    }

    async function refresh() {
        if (refreshing) { refreshAgain = true; return; }
        refreshing = true;
        refreshButton.disabled = true;
        status.textContent = "Loading annotation counts…";
        try {
            rows = await provider.getAnnotationSummary();
            render();
            status.textContent = `Counts updated ${new Date().toLocaleTimeString()}. Most annotations appear first by default; ties are alphabetical.`;
        } catch {
            status.textContent = rows.length
                ? "Could not refresh counts. Previous counts are still shown; use Refresh counts to retry."
                : "Could not load counts. Use Refresh counts to retry. You can still select a species below.";
        } finally {
            refreshing = false;
            refreshButton.disabled = false;
            if (refreshAgain) {
                refreshAgain = false;
                void refresh();
            }
        }
    }

    refreshButton.addEventListener("click", refresh);
    table.querySelectorAll("thead button").forEach((button) => {
        button.addEventListener("click", () => {
            descending = key === button.dataset.sort ? !descending : false;
            key = button.dataset.sort;
            render();
        });
    });
    return {
        refresh,
        updateLocks() {
            table.querySelectorAll(".summary-species").forEach((button) => { button.disabled = locked(); });
        },
    };
}
