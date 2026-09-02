-- Widget rows materialized from docs/PEEC_FEM_COMPARISON_RESULTS.json
-- (schema electrical-peec-fem-comparison/v2, measured 2026-09-02T08:25:56+0900).  The benchmark JSON
-- remains the canonical raw result; this normalized query is the report
-- renderer's reproducible source layer.
--
-- skin_error rows are FEM: relative error of the AC/DC resistance ratio against
-- the closed-form slab, by elements through the thickness.  skin_error_peec rows
-- are the sheet-PEEC filament solve of the same slab: the same error by bar
-- length in cells, with CPU and CUDA solve time and static operator size.
WITH widget_rows(
    dataset_id,
    sort_order,
    case_label,
    method,
    elements,
    unknowns,
    median_ms,
    relative_error,
    relative_error_percent,
    cpu_ms,
    cuda_ms,
    operator_kib,
    capability,
    sheet_peec,
    matrix_free_fem,
    comparison
) AS (
    VALUES
        ('dc_cpu_timing', 1, 'PEEC · 32',  'Sheet PEEC',       32,  33,  0.938098, 3.9619916976e-15, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
        ('dc_cpu_timing', 2, 'FEM · 32',   'matrix-free FEM',  32,  66,  4.560738, 2.1130622387e-12, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
        ('dc_cpu_timing', 3, 'PEEC · 64',  'Sheet PEEC',       64,  65,  0.951653, 4.1600912825e-14, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
        ('dc_cpu_timing', 4, 'FEM · 64',   'matrix-free FEM',  64, 130,  8.147450, 3.0899133028e-12, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
        ('dc_cpu_timing', 5, 'PEEC · 128',  'Sheet PEEC',      128, 129,  1.056970, 7.7258838103e-14, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
        ('dc_cpu_timing', 6, 'FEM · 128',   'matrix-free FEM', 128, 258, 16.812686, 5.5498699257e-12, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
        ('skin_error', 1, 'FEM · 16', 'matrix-free FEM',  16,   34,  10.851076, NULL, 1.8139531440, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
        ('skin_error', 2, 'FEM · 32', 'matrix-free FEM',  32,   66,  21.410566, NULL, 0.4637134125, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
        ('skin_error', 3, 'FEM · 64', 'matrix-free FEM',  64,  130,  47.532752, NULL, 0.1165505196, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
        ('skin_error', 4, 'FEM · 128', 'matrix-free FEM', 128,  258, 210.594103, NULL, 0.0291762639, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
        ('skin_error_peec', 1, 'PEEC · 16', 'Sheet PEEC',  16, 1152, 3694.709086, NULL, 11.7814123039, 3694.709086, 22422.536782, 535.500, NULL, NULL, NULL, NULL),
        ('skin_error_peec', 2, 'PEEC · 32', 'Sheet PEEC',  32, 2304, 11453.906802, NULL, 3.2121649078, 11453.906802, 53951.478227, 1039.500, NULL, NULL, NULL, NULL),
        ('skin_error_peec', 3, 'PEEC · 64', 'Sheet PEEC',  64, 4608, 94797.362345, NULL, 0.6841603355, 94797.362345, 238095.718400, 2047.500, NULL, NULL, NULL, NULL),
        ('dc_audit', 1, NULL, 'Sheet PEEC',       32,  33, NULL, 3.9619916976e-15, NULL,  0.938098,   5.555592, 2.125, NULL, NULL, NULL, NULL),
        ('dc_audit', 2, NULL, 'matrix-free FEM',  32,  66, NULL, 2.1130622387e-12, NULL,  4.560738, 111.056915, 0.510, NULL, NULL, NULL, NULL),
        ('dc_audit', 3, NULL, 'Sheet PEEC',       64,  65, NULL, 4.1600912825e-14, NULL,  0.951653,   6.521729, 4.125, NULL, NULL, NULL, NULL),
        ('dc_audit', 4, NULL, 'matrix-free FEM',  64, 130, NULL, 3.0899133028e-12, NULL,  8.147450, 269.543340, 0.947, NULL, NULL, NULL, NULL),
        ('dc_audit', 5, NULL, 'Sheet PEEC',      128, 129, NULL, 7.7258838103e-14, NULL,  1.056970,   7.038339, 8.125, NULL, NULL, NULL, NULL),
        ('dc_audit', 6, NULL, 'matrix-free FEM', 128, 258, NULL, 5.5498699257e-12, NULL, 16.812686, 489.681025, 1.822, NULL, NULL, NULL, NULL),
        ('capabilities', 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'DC PCB conduction / vias', '対応', '対応', '直接比較可能'),
        ('capabilities', 2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, '準静的部分インダクタンス', '対応', 'scalar Ezでは対象外', '適用範囲が異なる'),
        ('capabilities', 3, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, '表皮・近接効果', 'graded filamentで求解', '導体断面場を求解', '同じ平板閉形式へ双方が収束、離散化と問題規模が異なる'),
        ('capabilities', 4, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, '誘電損失・波動伝搬', '非対応', '2D scalar full-wave', 'FEMのみ'),
        ('capabilities', 5, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, '任意3D vector full-wave', '非対応', '非対応', '将来の辺要素が必要')
)
SELECT *
FROM widget_rows
ORDER BY dataset_id, sort_order;
