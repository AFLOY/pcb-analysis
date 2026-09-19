# pcb-analysis 作業規則

## 正本と採用状態

- 設計の正本は`docs/DESIGN.md`と`docs/REQUIREMENTS.md`、各packageの契約は
  `docs/MATRIX_FREE_MPIR_FEM.md`、`docs/SHEET_PEEC.md`、`docs/THERMAL_MPIR_FEM.md`、
  `docs/EMC_DIPOLE_SUPERPOSITION.md`、`docs/MULTIPHYSICS_SCENARIOS.md`、
  `docs/GEOMETRY_CAD_IMPORT.md`、`docs/VOXEL_PEEC.md`である。
- 採用判定はbenchmark dataで行う。同じfixture、同じ精度要求、同じdeviceで現行実装と
  候補を突き合わせ、solve時間、inner/outer反復数、到達残差、解の差、storageを記録する。
  単体testの合格、画像、個別指標の改善だけで採用方式を変更しない。
- 採用した比較の要約(数値表と判定)は対象packageの`docs/*.md`へ昇格する。不採用の候補も
  「measured, not adopted」として同じ文書へ要約を残し、実装は`exp/`branchとtagに留める。
- 研究計画や旧実装は`../plane_opt`、`../plane_opt_refactor`のGit履歴を参照し、この
  リポジトリへ複製しない。

## 安定した責務境界

- `src/electrical/sheet_peec/`: 2.5D sheet PEEC。sheet mesh・演算子・CUDA solve、板厚方向の
  skin filamentsと厚さ判定、`plane_opt`向け契約(`plane_opt_contract`)。
- `src/electrical/dice_peec/`: DICEのdelta scoring(2.5D多層proxy)、router向け`layout_ops`、
  runtime controllerとbackend、`Stackup`、CLI。電流や電位は解かない。
- `src/electrical/voxel_peec/`: PyPEECの3D voxel PEEC。配列契約(`contract`)、CUDA実行方針と
  計測(`cuda_pypeec`)、memory予測(`pypeec_memory`)。PyPEECが組立てとsolveを所有する。
  計測型2つを`dice_peec.controller`からimportする以外に`electrical`内の依存は持たない。
- `src/electrical/matrix_free_mpir_fem/`: MPIR solverとNumPy/CuPy runtime、Q1 DC伝導、
  2D周波数領域Maxwell、fused CUDA kernel。solverとruntimeはここが唯一の所有者である。
- `src/thermal/matrix_free_mpir_fem/`: 定常熱伝導の離散化、対流・輻射(Newton線形化)境界、
  二段preconditioner、CUDA kernel、電気→熱のJoule loss写像。solverとruntimeは`electrical`からimportし複製しない。
- `src/emc/tiled_dipole_superposition/`: 電流分布を読むadapter、dipole場、遠方界、規格
  limit。場のsolverを持たない。
- `src/multiphysics/staggered_coupling/`: 上記を連結するscenarioだけを置く。物理の離散化
  や新しいsolverをここへ書かない。
- `src/geometry/cad_import/`: STEP(OpenCASCADE、`OCP`)の読込、2.5D板の断面
  ラスタライズ、3D物体のvoxel化、板とvoxelの接触map。solverを持たず、`OCP`は
  `reader.py`だけがimportし、公開結果は配列とdataclassに限る。

importは`electrical` ← `thermal` ← `multiphysics`、`electrical` ← `emc` ←
`multiphysics`、`electrical`/`thermal` ← `geometry` ← `multiphysics`の一方向であり、
`electrical`は他packageをimportしない。package境界を越えて
`_`始まりの名前をimportしない。新しい物理は`src/<physics>/<method+acceleration>/`へ置き、
既存packageの中へ足さない。

CLIは`electrical.dice_peec.cli`に集める。console scriptを`pyproject.toml`へ追加したら
CIの「Check what the distribution actually ships」で起動確認する。`py.typed`は各top-level
packageに置き、`[tool.setuptools.package-data]`へ登録する。

## データと成果物

- `experiments/`は既定でGit管理外である。再実行可能な受入れ・比較toolだけを
  `.gitignore`の`!experiments/<name>.py`で個別に追跡する。一度限りのprobeは追跡しない。
- 実験の生の出力は`benchmark-results/`(Git管理外)へ書く。文書が参照する監査結果は
  `docs/*_RESULTS.json`、`docs/*_ARTIFACT.json`、`docs/*_REPORT.html`として追跡し、測定
  環境(device、CuPy、NumPy、日時)を含める。
- benchmark scriptの出力JSONには`decision`(採用/不採用)と`environment`を含める。
  文書の表はこのJSONから転記し、JSONに無い数値を書かない。
- 生成物、`dist/`、`build/`、cache、`.venv/`、認証情報をcommitしない。
- agentの作業note、handoffは`.agents/`へ置きGit管理外とする。経緯はcommit本文とtagが持つ。

## branchと実験

- 安定変更は`feature/<purpose>`、修正は`fix/<purpose>`、component比較は
  `exp/<purpose>`、文書だけなら`docs/<purpose>`、CI/配布は`ci/<purpose>`を使う。
- 実験のsourceは同じ`src/`をbranch上で編集し、別directoryへ複製しない。
- `exp/`branchをそのままmainへmergeしない。採用する一般実装は`feature/`へ整理し、既存の
  全testと該当packageのCUDA testを通す。不採用の`exp/`branchはpushしてtagを打ち、削除も
  mergeもしない。
- mainへはfast-forwardで入れ、merge commitを残さない。PRを使う場合もrebaseして
  fast-forwardまたはrebase mergeにする。
- merge済みのbranchを削除しない。作業単位の境界を示すのはbranchとtagだけである。

## 検証とGit

- 作業開始時に`git status`、HEAD、`origin/main`との差を確認する。localのmainは
  `origin/main`へ合わせてから分岐する。
- Python変更は`.venv/bin/python -m pytest tests/ -q`を通す。CUDAを触る変更はGPUのある
  shellでCUDA testを実行し、deviceとCuPy版をtagまたはcommit本文へ記録する。skipされた
  CUDA testを合格として記録しない。
- packaging、`pyproject.toml`、CIを触る変更はlocalで`python -m build`と
  `twine check --strict`を通し、clean venvへwheelをinstallして全packageをimportする。
- 実行しなかったtest、benchmark、gateを成功として記録しない。
- 1 commitは1論理変更とし、`feat:`、`fix:`、`perf:`、`test:`、`docs:`、`chore:`、`ci:`、
  `exp:`を使う。1つの作業branchでも、solver hook、新package、packaging、CIは別commitに
  分ける。
- 作業単位の境界へ注釈付きtag`work/<purpose>`を打つ。本文へ対象commit範囲、検証したこと、
  採用/不採用の根拠となる数値、検証していないことを書く。branch名と同名のtagは打たない。
- GitHubへpushする前に、同じshellで`source ~/.bash_profile`を実行し`GITHUB_PASS`を読込む。
  未設定または空ならpushせず、認証不足として報告する。`GITHUB_PASS`は`GIT_ASKPASS`または
  credential helperへ環境変数として渡し、remote URL、Git設定、command引数、log、commit、
  tag本文へ保存または表示しない。shell traceを有効にしない。
- pushはmain、作業branch、`work/`tagを含める。force pushをmainへ行わない。

## リリース

- 版は`pyproject.toml`の`version`だけが持つ。公開する変更は`chore: release X.Y.Z`で
  版を上げ、mainへfast-forwardしてから`vX.Y.Z`のGitHub releaseを公開する。tagと版が
  一致しないと`Release` workflowが失敗する。
- `Release` workflowはsetuptools 69でbuild、`twine check --strict`、clean venvへの
  install、PyPIへのTrusted Publishing(OIDC)、GitHub releaseへの資産添付を行う。API token
  をrepositoryへ置かない。
- 本番公開の前に`Release`を`workflow_dispatch`、target `testpypi`で手動実行し、
  TestPyPIへ上がることを確認する。
- 新しいtop-level packageや破壊的なimport pathの変更はminor版を上げ、READMEの移行noteを
  更新する。
