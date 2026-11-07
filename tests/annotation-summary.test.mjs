import test from "node:test";
import assert from "node:assert/strict";
import { ServerProvider } from "../docs/annotation-data.mjs";
import { sortSummary } from "../docs/annotation-summary.mjs";

test("summary defaults to most votes with alphabetical ties, supports every column, and preserves input", () => {
    const rows = [
        { scientificName: "B", counts: { positive: 2, negative: 1, unsure: 3, total: 6 } },
        { scientificName: "C", counts: { positive: 0, negative: 0, unsure: 0, total: 0 } },
        { scientificName: "A", counts: { positive: 1, negative: 3, unsure: 2, total: 6 } },
    ];
    const names = (items) => items.map((item) => item.scientificName).join("");
    assert.equal(names(sortSummary(rows)), "ABC");
    assert.equal(names(sortSummary(rows, "total", false)), "CAB");
    assert.equal(names(sortSummary(rows, "positive", true)), "BAC");
    assert.equal(names(sortSummary(rows, "negative", true)), "ABC");
    assert.equal(names(sortSummary(rows, "unsure", true)), "BAC");
    assert.equal(names(sortSummary(rows, "scientificName")), "ABC");
    assert.equal(names(rows), "BCA");
});

test("summary always fetches fresh counts for all species and rejects invalid counts", async () => {
    let count = 0;
    let counts = { positive: 0, negative: 0, unsure: 0, total: 0 };
    const provider = new ServerProvider("https://example.com", null, {
        request: async (path) => {
            assert.equal(path, "/species");
            count++;
            return { ok: true, json: async () => ({ species: [{
                species_id: "FROG", scientific_name: "Frog", common_name: "Frog",
                annotation_counts: counts,
            }] }) };
        },
    });
    assert.equal((await provider.getAnnotationSummary())[0].counts.total, 0);
    counts = { positive: 1, negative: 2, unsure: 3, total: 6 };
    assert.equal((await provider.getAnnotationSummary())[0].counts.total, 6);
    assert.equal(count, 2);
    for (const invalid of [undefined, { ...counts, positive: -1 }, { ...counts, total: 5 }, { ...counts, unsure: "3" }]) {
        counts = invalid;
        await assert.rejects(provider.getAnnotationSummary(), /Invalid annotation counts/);
    }
});
