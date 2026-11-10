import test from "node:test";
import assert from "node:assert/strict";
import { annotatorLabel, filterRows, sortRows, pageRows, resultCounts } from "../osa_anurans_app/browser/browser.mjs";

const rows = [
    { row_id: "1", window_id: "clip-a", species_id: "FROG", response: "present", annotator_name: "Alice", annotator_id: "session-1", created_at: "2026-11-01T00:00:00Z", recording_filename: "Forest.wav" },
    { row_id: "2", window_id: "clip-a", species_id: "FROG", response: "absent", annotator_name: "Alice", annotator_id: "session-2", created_at: "2026-11-02T00:00:00Z" },
    { row_id: "3", window_id: "clip-b", species_id: "TOAD", response: "unsure", annotator_name: "Bob", created_at: null },
];

test("filters each answer and groups annotator names across sessions", () => {
    assert.deepEqual(filterRows(rows, { annotator: "Alice" }), rows.slice(0, 2));
    assert.deepEqual(filterRows(rows, { annotator: "Alice", response: "absent" }), [rows[1]]);
    assert.deepEqual(filterRows(rows, { species: "TOAD", response: "present" }), []);
    assert.deepEqual(filterRows(rows, { search: " FOREST " }), [rows[0]]);
    assert.deepEqual(filterRows(rows, { search: "CLIP-A" }), rows.slice(0, 2));
});

test("counts annotations independently from unique clips", () => {
    assert.deepEqual(resultCounts(rows), { annotations: 3, clips: 2 });
    assert.deepEqual(resultCounts([...rows, rows[0]]), { annotations: 4, clips: 2 });
    assert.deepEqual(resultCounts([]), { annotations: 0, clips: 0 });
});

test("sorts newest first, retains duplicates, puts unavailable dates last", () => {
    assert.deepEqual(sortRows(rows).map((row) => row.row_id), ["2", "1", "3"]);
    const duplicate = { ...rows[0], row_id: "4" };
    assert.deepEqual(sortRows([duplicate, rows[0]]).map((row) => row.row_id), ["1", "4"]);
    assert.equal(rows[0].row_id, "1");
});

test("pagination preserves row boundaries and missing names are explicit", () => {
    assert.deepEqual(pageRows(rows, 1, 2), rows.slice(0, 2));
    assert.deepEqual(pageRows(rows, 2, 2), [rows[2]]);
    assert.equal(annotatorLabel({ annotator_name: "   " }), "Unavailable");
});
