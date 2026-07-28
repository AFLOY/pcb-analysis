# peec-fastopt 要求仕様

PCB の銅面最適化が場のソルバに何を要求するかを述べる。要求元は
`plane_opt_refactor` の採用処理列だが、要求そのものは特定の利用者に閉じない形で
書く。

各項目には**検証済みか否か**を明記する。未検証の要求を満たしたことにはしない。

---

## 1. 契約

### 1.1 求解単位

1 回の求解は「1 つの導体形状、1 つの role、1 組の端子電流、1 つの周波数」である。
role とは同一電位に属する銅の集合で、他の role の銅はこの求解の導体ではない。

### 1.2 入力

| 要素 | 内容 |
|---|---|
| 導体 | セル格子上の占有。層ごとに 1 面。層は stackup が述べる高さに置く |
| 層間接続 | ビアと貫通穴。どのセルのどの層対を繋ぐか |
| 端子 | pad ごとのセル集合と注入電流。総和はゼロ |
| 材料 | 銅の抵抗率、層ごとの厚み |
| 周波数 | 0 を含む |

導体形状は最適化の反復ごとに変わる。stackup と格子は変わらない。**同一 epoch 内で
形状だけが変わる求解列**が典型的な使い方であり、形状に依らない準備物の再利用が
要求に含まれる（§4.3）。

### 1.3 出力

| 要素 | 単位 | 用途 |
|---|---|---|
| 節点電位 | V（複素） | 電圧スパン |
| 枝電流 | A（複素） | 電流密度、層間電流、ジュール損 |
| 収束情報 | 反復数、相対残差、収束判定 | ゲートが「評価されていない」を検出するため |

**電流密度は面内分布として要求する。** 最適化は密度分布を整形の駆動に使うので、
集約値だけでは足りない。

### 1.4 座標の同一性

出力は入力と同じセル索引で返さなければならない。ソルバ内部で境界箱を切り詰める
場合、逆写像は層込みで正しく戻す必要がある。層を落として面内座標だけで返してはな
らない。

---

## 2. 指標

### 2.1 ゲートが読む指標

これらは合否判定に直接入る。欠けた場合、ゲートは判定不能として拒否しなければなら
ない。

| 指標 | 定義 | 備考 |
|---|---|---|
| `bulk_p99_current_density_a_per_mm2` | 端子セルを除いた面内電流密度の 99 分位 | 端子を除くのは、注入点の特異性を形状評価に混ぜないため。セル1つの密度の定め方は backend ごとに違う（§2.5） |
| `voltage_span_v` | DC: 実部の最大−最小 / AC: 参照節点からの位相子振幅の最大 | 2 つの定義があるので、どちらを使ったかを併記すること（§2.3） |
| `max_vertical_connection_current_a` | 層間を渡る電流の最大 | ベクトルの長さではなく z 成分から取ること。接合部の面内成分を層移動の電流として数えてはならない |
| `current_closure_error_a` | 要求電流と実現電流の差の最大 | KCL の健全性 |

### 2.2 記録に要る指標

| 指標 | 用途 |
|---|---|
| `i2r_loss_w` | 比較と回帰検出 |
| `magnetic_energy_j` | ループインダクタンスの代理 |
| `max_current_density_a_per_mm2` | 分位との乖離の把握 |
| `conductive_node_count` | 導体が期待通り立っているかの検算 |

### 2.3 端子の模型

端子はセル集合へ電流を配分する。その集合を等電位にはしない。PyPEEC の lumped 電流源も
同じで、領域の voxel へ電流を配分しつつ電位は独立変数のまま残す。両者の規約は一致して
いる。

この選択は結果を左右する。等電位電極なら導体が自分で交流分布を決めるが、電流密度を
指定する境界では、指定した分布から発達分布へ緩和する助走距離が要る。助走距離はおよそ
最大横寸法 ÷ π で、3.5mm 厚のバーなら約 1.1mm である。端子から近い場所の値はこれに
汚染されるので、測定窓は端から助走距離以上離さなければならない。

### 2.4 定義の明示が要求である

同じ名前で違う量を返してはならない。`voltage_span_v` は DC と AC で定義が異なる
ため、`voltage_span_definition` として `signed_max_minus_min` か
`maximum_phasor_magnitude_from_reference` のいずれかを併記することを要求する。

断面が等電位でない場合、**節点電位の算術平均は抵抗の観測量ではない**。抵抗を報告
する場合はジュール損 `Σ R_b |I_b|² / |I|²` から取ること。密な相互インダクタンスを
部分領域だけ切り出した `Re(I^H Z I)` を使ってはならない。領域間の無効電力交換が実
部に現れる。部分領域の損失には対角抵抗だけを使う。

### 2.5 セル密度の規約は backend ごとに違う

| backend | セルの密度 |
|---|---|
| `resistive_mesh` | 接続する面内枝の電流の最大 |
| `pypeec` / `cuda_peec` | 電流密度ベクトルの大きさ |
| `sheet_peec` | 両側の枝の平均を軸ごとに取り、2軸を大きさとして合成 |

一様な帯導体では 3 つとも一致する。分かれるのは導体の縁と角で、power_module の PGND
で実測すると `resistive_mesh` と `sheet_peec` は最大 19.7%、平均 8.3% 違う（相関 0.993）。

接続枝の最大は、大きさでも 1 成分でもなく、いずれか 1 成分の上界である。物理量として
正しいのは大きさの側で、`sheet_peec` は PEEC 経路の規約に揃えてある。`resistive_mesh`
を揃えるのは採用方式の変更であり、記録済みの全ゲート結果が動くため、ここでは差を記録
するにとどめる。

なお同じ導体・同じ case で `voltage_span_v` と `i2r_loss_w` は `resistive_mesh` と
`sheet_peec` で 5e-12 および 3e-14 まで一致する。節点電位は接地点が違うぶん定数だけ
ずれ、その定数を除くとスパンの 1.4e-10 で一致する。密度だけが規約で分かれている。

---

## 3. 精度要求

### 3.1 何に対する精度か

ゲートは**比**を読む。候補形状の指標を、同じ導体を全領域に敷いた基準の指標で割っ
た値である。したがって系統誤差の一部は分子と分母で相殺する。ただし相殺を根拠に
誤差を無視してはならない。候補と基準では電流分布が違うので、分布に依存する誤差は
相殺しない。

### 3.2 直流

DC では誘導が落ち、残るのは純粋な抵抗網である。**独立に組んだ抵抗網と機械精度で
一致することを要求する。** ここが合わなければ、接続行列・抵抗・ソース・接地の
いずれかが誤っている。誘導の問題と分離して検出できる唯一の点である。

実測: `1.6e-19`（スケール `7.2e-4`）。要求を満たす。

### 3.3 交流、薄銅

35µm 箔は 300kHz で `t/δ = 0.29`、つまり厚み方向に一様である。1 フィラメントで足
り、厚み方向の離散化誤差は無視できる。この前提が崩れるのは約 0.9MHz からで、それ
以上を見る場合は §3.4 が箔にも適用される。

### 3.4 交流、厚銅

銅インレイのような厚い導体では表皮効果が支配的になる。

| 周波数 | δ | 3.5mm インレイ `t/δ` | 広幅平板 R_ac/R_dc |
|---|---|---|---|
| 1 kHz | 2.090 mm | 1.67 | 1.04 |
| 100 kHz | 0.209 mm | 16.75 | 8.37 |
| 300 kHz | 0.121 mm | 29.01 | 14.50 |
| 1 MHz | 0.066 mm | 52.96 | 26.48 |

3.5mm インレイが厚み方向に一様と言えるのは **89Hz まで**である。300kHz では
1 フィラメントで扱えば交流抵抗を 14.5 倍過小評価する。

**要求は解像度で述べる。** 独立な場の解（1.6×3.5mm 銅バー断面、300kHz、無限長、
`Rac/Rdc = 5.92 ± 0.03`）に対し、厳密な電流分布を与えたときに離散化が保持できる
損失の割合を実測した。ソルバの収束とは無関係な天井である。

| 面内 pitch / δ | δ あたりフィラメント数 | 保持率 |
|---|---|---|
| 1.66 | 2 | 77.2% |
| 1.66 | 4 | 84.0% |
| 1.66 | 8 | 89.9% |
| 0.83 | 2 | 87.9% |
| 0.83 | 4 | 94.8% |
| 0.41 | 4 | 98.4% |

**面内と厚み方向のコストはほぼ等しく、どちらか一方だけでは届かない。** 95% に達す
るには pitch ≲ 0.83δ かつ δ あたり 4 フィラメントの両方が必要である。

0.41 で 8 フィラメントの点が 1.045 になるのは、そこでは sheet セルが場の解自身の
25µm より細かくなり比較が飽和したものである。優位性ではない。

### 3.5 精度要求は解像度の選択として提示すること

ソルバは、与えられた解像度で何割の精度が期待できるかを**呼び出し前に**報告できな
ければならない。利用者が後から原因不明の不足として発見する状態は要求違反とする。

`peec_fastopt.skin_filaments.discretisation_note()` がこれを担う。

---

## 4. 計算資源

### 4.1 事前判定

入らない求解は、確保を始める前に拒否できなければならない。実行して途中で落ちる形
は要求違反とする。理由は多層化で桁が変わることである。

| 34.7mm 角、両面基板、0.2mm pitch | prepared operators |
|---|---|
| 1 層 | 4.5 MiB |
| 2 層（3次元 voxel、nz=45） | 541.1 MiB |
| 2 層（2.5次元 sheet） | 5.6 MiB |

導体は約 2 倍にしかならないのに、3次元 voxel の演算子は 120 倍になる。1,108,080
voxel の箱に導体は 29,006 しかない。

3.5mm インレイを入れると 300kHz で:

| | prepared operators |
|---|---|
| 3次元 voxel、dz=60µm（35µm 箔を表現できない） | 0.95 GiB |
| 3次元 voxel、dz=35µm、nz=114 | 1.65 GiB |
| 2.5次元 sheet、フィラメント 11 本 | 122.7 MiB |

`peec_fastopt.pypeec_memory.estimate_pypeec_memory()` が 3次元経路の事前判定を
担う。計上できない量（cuFFT の workspace）は明示した係数として分離すること。無言
で丸め込んではならない。

### 4.2 設定は実効を伴うこと

適用されない設定を受け付けてはならない。`precision` に `complex64` を受け付けな
がら complex128 で解いていた状態は要求違反であり、拒否に変更した。実現できない
要求は受け付けないか、`precision_request_honored` として満たせなかったことを報告
する。

### 4.3 epoch を跨ぐ再利用

形状に依らない準備物（Green テンソル、FFT plan、メッシュ）は、同一 epoch の求解間
で保持されなければならない。求解ごとに executor を作り直して pool と plan を捨て
る形は、薄い模型では小さな浪費だが、基板の高さを張る模型では小さくない。

現状の制約: PyPEEC は operator 準備と sweep 求解の間に接合点を公開していないため、
**呼び出しを跨ぐ operator 再利用はできない**。1 回の呼び出し内では sweep ループの
前に一度だけ構築されるので再利用されるが、両基板とも全 case の端子 pad 集合が相異
なるため、sweep への統合には統合対象がない。

---

## 5. 検証要求

主張には参照解を伴わなければならない。参照解の種類ごとに、何を切り分けられるかが
異なる。

| 参照解 | 切り分けられるもの | 実測 |
|---|---|---|
| 独立に組んだ抵抗網（DC） | 接続行列、抵抗、ソース、接地 | 1.6e-19 |
| 同じ演算子の密行列直接解 | 反復解法と FFT 経路 | 4.2e-11（300kHz） |
| 既知の閉形式（Grover、フィラメント式） | 部分インダクタンス核 | 0.006% |
| 断面の場の解（FEM/FD、無限長） | 表皮効果の絶対値、端部効果なし | §3.4 |
| 場の解の分布を投影して Z を直接当てる | 離散化の天井（ソルバと端子を除外） | §3.4 |

### 5.1 検証済み

| 項目 | 結果 |
|---|---|
| 自己部分インダクタンス vs Grover 式（比 10〜1000） | 0.007〜0.17% |
| 相互部分インダクタンス vs 厳密フィラメント式 | 0.0002〜0.006% |
| FFT 演算子 vs 同じ閉形式の直接和 | 3.2e-11 |
| 演算子の対称正定値性 | 成立 |
| 直交枝の相互インダクタンス | 厳密にゼロ |
| DC vs 独立な抵抗網 | 1.6e-19 / スケール 7.2e-4 |
| AC vs 密行列直接解（0 / 3e5 / 1e8 Hz） | 1.2e-14 / 4.2e-11 / 3.7e-11 |
| 抵抗解からの乖離の周波数依存 | 一次、小数 3 桁まで |
| 発達分布に対する軸方向電界の共通性 | 最悪 6.1%、支配的な虚部で 3.7% |

### 5.2 検証していないこと

- **PyPEEC はこの環境で一度も実行されていない**（未インストール）。記録されている
  どの最適化実行も `backend: numpy`、すなわち抵抗網経路である。3次元経路について
  確かめたのは mesher へ渡す入力の構造と、solution からの読み戻しだけで、その間の
  求解は含まない。
- **端点間の sheet 求解は絶対値として検証できていない。** 長さを振ると、中央 1.6mm
  の固定窓で測ったジュール損の比は 2.774 / 4.155 / 4.998 / 5.313（長さ 2.4 / 4.8 /
  9.6 / 19.2mm）と上昇し、19.2mm でもまだ上昇中である。場の解の 5.92 に対し 90% だが、
  長さ方向に収束していないので、この 90% を精度として扱うことはできない。上昇の傾向は
  助走距離（§2.3、約 1.1mm）と整合する。
- **射影の天井をソルバ出力の上界として使ってはならない。** 一度そう扱ったが誤りである。
  天井は「真の分布をこの mesh 上で表現したときに保持できる損失」であり、ソルバは真の
  分布を再現する義務を負わない。面内格子が側壁の表皮層を解像できない以上、離散解は真の
  分布と違う集中の仕方をし、この mesh 上でより大きい損失を持ち得る。実測でも長さ 9.6mm
  以上で天井 4.93 を超える。天井が束縛するのは mesh の表現力であって求解の出力ではない。
- CUDA 上では何も動かしていない。変換は `numpy.fft` である。

---

## 6. 定式化と離散化の理論

### 6.1 適用する場の近似

本ソルバの sheet 経路は、導体中の導電電流とその磁気結合を解く**磁気準静的 PEEC** である。変位電流、誘電体の分極、電磁波の遅延は含まない。したがって、以下の定式化が成立する範囲は、

1. 各電流フィラメント内で電流密度を一定と近似できること
2. 導体を、面内格子と少数の既知の高さに置かれたフィラメントで表せること
3. 容量性電流と伝搬遅延が導電電流分布を有意に変えないこと

である。

### 6.2 Ruehli の部分インダクタンス

電流方向が一定の直方体要素 \(i,j\) を考える。電流方向の単位ベクトルを
\(\hat{\boldsymbol l}_i,\hat{\boldsymbol l}_j\)、電流に直交する断面積を
\(a_i,a_j\)、体積を \(V_i,V_j\) とすると、部分相互インダクタンスは

\[
L^{\mathrm p}_{ij}
=
\frac{\mu}{4\pi a_i a_j}
\int_{V_i}\int_{V_j}
\frac{
\hat{\boldsymbol l}_i\cdot\hat{\boldsymbol l}_j
}{
\lVert\boldsymbol r-\boldsymbol r'\rVert
}
\,dV'\,dV
\]

である。FR-4 と銅を非磁性とする本モデルでは \(\mu=\mu_0\) である。実装は
`peec_fastopt/sheet_inductance.py` の `closed_form_mutual_inductance()` と
`self_partial_inductance()` に対応する。この定義は Ruehli の PEEC 導出に基づく。[^ruehli]

通常のループインダクタンスは、往路と復路を一組にしなければ定義できない。一方、部分インダクタンスは各導体片が作るベクトルポテンシャルへの寄与を、復路を仮定せずに保持する。枝の向きを表すループベクトルを \(\boldsymbol c\) とすれば、閉ループのインダクタンスは後から

\[
L_{\mathrm{loop}}
=
\boldsymbol c^{\mathsf T}
\boldsymbol L^{\mathrm p}
\boldsymbol c
=
\sum_{i,j}c_i c_j L^{\mathrm p}_{ij}
\]

として得られる。この分解により、同じ部分要素を異なる端子条件や導体形状で再利用でき、復路が最適化中に変化しても演算子自体を作り直す必要がない。部分インダクタンス単体は端子間で直接測る量ではなく、自己項と全相互項を組み合わせて初めて物理的なループ量になる。

### 6.3 sheet mesh と回路方程式

各占有セルを一つの節点とし、同一層で隣接する占有セル間に \(x\) または \(y\) 方向の枝を置く。枝が表す直方体は、長さが pitch、面内幅も pitch、厚さがその層の銅厚である。層間接続とフィラメント間接続は \(z\) 方向の枝として追加する。実装は `SheetMesh.__post_init__()`、`SheetMesh.incidence()` にある。

枝–節点接続行列を \(\boldsymbol A\)、枝電流を \(\boldsymbol I\)、節点電位を
\(\boldsymbol V\)、節点への注入電流を \(\boldsymbol I_{\mathrm{src}}\) とすると、

\[
\boldsymbol Z\boldsymbol I-\boldsymbol A\boldsymbol V=0,
\qquad
\boldsymbol A^{\mathsf T}\boldsymbol I
=
\boldsymbol I_{\mathrm{src}},
\]

\[
\boldsymbol Z
=
\boldsymbol R+j\omega\boldsymbol L^{\mathrm p}
\]

を解く。第1式は各枝の電圧降下、第2式は節点の KCL である。
`SheetMesh.incidence()` は、枝の始点に \(+1\)、終点に \(-1\) を置くため、
\(\boldsymbol A\boldsymbol V\) が枝方向の電位差になる。

\(\boldsymbol A\boldsymbol 1=0\) なので、全節点電位へ同じ定数を加えても枝電圧は変わらない。連結成分ごとに一つのゲージ条件が必要である。現行の
`solve_sheet_case()` は節点 0 を \(0\ \mathrm V\) に固定する。これは導体が一つの連結成分である場合に限り十分であり、複数成分の扱いは §7 の未解決課題である。

枝電流を消去すれば、形式上は

\[
\boldsymbol I=\boldsymbol Z^{-1}\boldsymbol A\boldsymbol V,
\qquad
\boldsymbol A^{\mathsf T}
\boldsymbol Z^{-1}
\boldsymbol A\boldsymbol V
=
\boldsymbol I_{\mathrm{src}}
\]

という節点アドミタンス系になる。DC では \(\boldsymbol Z=\boldsymbol R\) が対角なので、この消去は安価であり、現行実装も
`reduced.T @ diag(1 / resistance) @ reduced` を直接解く。

交流では \(\boldsymbol L^{\mathrm p}\) が各同軸方向内で密である。明示的な逆行列は、枝数を \(B\) として概ね \(O(B^3)\) の準備と \(O(B^2)\) の記憶を要する。行列を形成しない場合でも、外側の節点 Krylov 反復の各積ごとに
\(\boldsymbol Z^{-1}\) の内側 Krylov 求解が必要となり、反復コストが乗算される。このため現行の交流経路は電流を消去せず、

\[
\begin{bmatrix}
\boldsymbol Z & -\boldsymbol A\\
\boldsymbol A^{\mathsf T} & 0
\end{bmatrix}
\begin{bmatrix}
\boldsymbol I\\
\boldsymbol V
\end{bmatrix}
=
\begin{bmatrix}
0\\
\boldsymbol I_{\mathrm{src}}
\end{bmatrix}
\]

を一つの GMRES で解き、反復ごとに \(\boldsymbol Z\) を一度適用する。
`solve_sheet_case()` 内の `impedance()`、`saddle()`、`precondition()` がこれに対応する。

### 6.4 直交枝間の結合がゼロになる理由

部分インダクタンス積分には

\[
\hat{\boldsymbol l}_i\cdot\hat{\boldsymbol l}_j
\]

が掛かる。したがって \(x\) 枝と \(y\) 枝では内積が厳密にゼロであり、相互部分インダクタンスも厳密にゼロになる。同様に、\(z\) 方向の縦枝は \(x,y\) のいずれとも結合しない。

この結果、面内演算子は

\[
\boldsymbol L^{\mathrm p}_{\parallel}
=
\begin{bmatrix}
\boldsymbol L_x & 0\\
0 & \boldsymbol L_y
\end{bmatrix}
\]

となり、\(x\) 電流用と \(y\) 電流用の二つの演算子だけを持てばよい。
`SheetInductanceOperator.apply()` も二方向を独立に変換する。\(x\) 電流から \(y\) 磁束が生じないことと演算子の正定値性は §5.1 の測定結果を正本とする。

ただし、縦枝同士は平行なので \(z\)-\(z\) 結合はゼロではない。現行実装がこれを持たないことは理論上の省略であり、§7.1 に記す。

### 6.5 並進不変性と二次元畳み込み

媒質の透磁率が一様なら、静的 Green 関数は

\[
G(\boldsymbol r,\boldsymbol r')
=
\frac{1}{\lVert\boldsymbol r-\boldsymbol r'\rVert}
=
G(\boldsymbol r-\boldsymbol r')
\]

であり、絶対位置には依存しない。同じ軸を向く二枝の幾何形状も各層内で共通なので、層 \(\ell,m\) 間の結合核は、面内セル位置を
\(\boldsymbol p,\boldsymbol q\) として

\[
K^{(\alpha)}_{\ell m}
(\boldsymbol p,\boldsymbol q)
=
K^{(\alpha)}_{\ell m}
(\boldsymbol p-\boldsymbol q;\,
z_\ell-z_m),
\qquad \alpha\in\{x,y\}
\]

となる。従って磁束鎖交は

\[
\Phi_{\alpha,\ell}(\boldsymbol p)
=
\sum_m\sum_{\boldsymbol q}
K^{(\alpha)}_{\ell m}
(\boldsymbol p-\boldsymbol q)
I_{\alpha,m}(\boldsymbol q)
\]

という層対ごとの二次元線形畳み込みである。導体形状は、非占有セルの電流をゼロにすることで入るため、核には入らない。

`build_kernel()` が符号付き面内 offset と層間距離から核を作り、
`SheetInductanceOperator.__init__()` が層対・軸ごとのスペクトルを準備する。
`2*rows × 2*cols` へのゼロ padding により、FFT の巡回畳み込みを物理的な線形畳み込みへ変換している。現在の変換は
`numpy.fft.rfft2()`／`irfft2()` である。

この構造は、一様な面内 pitch、一様な磁気媒質、同一層内で共通の枝形状を必要とする。局所細分化、位置依存の透磁率、傾斜導体を導入すると、そのままでは並進不変性を失う。

### 6.6 Hoer–Love 閉形式と桁落ち

二つの直方体に対する積分は、各空間軸について二回、合計六回積分した原始関数 \(F(x,y,z)\) を用いて評価できる。軸 \(u\) の中心差を
\(\Delta u\)、二つの直方体の寸法を \(a_u,b_u\) とし、

\[
u_{s_ut_u}
=
\Delta u
+
s_u\frac{a_u}{2}
+
t_u\frac{b_u}{2},
\qquad
s_u,t_u\in\{-1,+1\}
\]

と置けば、二重体積積分は

\[
\int_{V_i}\int_{V_j}\frac{dV'\,dV}{\lVert\boldsymbol r-\boldsymbol r'\rVert}
=
\sum_{\substack{s_x,t_x=\pm1\\s_y,t_y=\pm1\\s_z,t_z=\pm1}}
(s_xt_x)(s_yt_y)(s_zt_z)
F(x_{s_xt_x},y_{s_yt_y},z_{s_zt_z})
\]

となる。各軸に四つの corner difference があるため、項数は
\(4^3=64\) である。これは Hoer–Love の直方体導体の閉形式に対応する。[^hoer-love]
実装は `_primitive()` と `closed_form_mutual_inductance()` であり、自己項の
\(1/r\) 特異点も体積積分として可積分なので有限値を返す。

この式は厳密算術では正確だが、浮動小数点では遠距離で条件が悪化する。個々の
\(F\) は corner 座標のおよそ5乗で増加する一方、64項の符号付き和は実際の積分値程度にしかならない。そのため、残存率

\[
\eta_{\mathrm{sum}}
=
\frac{\left|\sum_k s_k F_k\right|}
{\max_k |F_k|}
\]

が小さいほど、相対丸め誤差は概ね
\(\epsilon_{\mathrm{mach}}/\eta_{\mathrm{sum}}\) に増幅される。

0.2 mm セルについて `closed_form_precision()` と `kernel_accuracy()` で測定した境界は次の通りである。測定の正本は `docs/SHEET_PEEC.md` の “The kernel” と
`tests/test_sheet_inductance.py` の `CancellationTests` に置く。

| 電流方向 offset | 符号付き和の残存率 | 中心間近似の相対誤差 |
|---:|---:|---:|
| 1 cell | \(6.3\times10^{-2}\) | \(1.0\times10^{-1}\) |
| 4 cells | \(1.5\times10^{-4}\) | \(5.1\times10^{-3}\) |
| 16 cells | \(8.1\times10^{-8}\) | \(3.2\times10^{-4}\) |
| 24 cells | \(7.8\times10^{-9}\) | \(1.4\times10^{-4}\) |
| 64 cells | \(2.5\times10^{-11}\) | — |
| 128 cells | \(4.0\times10^{-13}\) | — |

このため `mutual_partial_inductance()` は
`NEAR_RADIUS_CELLS = 24` 以内で閉形式を使い、それより遠方では

\[
L^{\mathrm p}_{ij}
\simeq
\frac{\mu_0}{4\pi}
\frac{l_i l_j}{r_{ij}}
\]

という中心間極限へ切り替える。24 cells では閉形式側にまだ約7桁が残り、中心間近似側も既に \(1.4\times10^{-4}\) 以内なので、切替点は両方式の破綻点から離れている。Grover 式、平行フィラメント式、密行列直接和との測定値は重複を避け、§5.1 を参照する。

### 6.7 表皮効果のフィラメント離散化

均一導体内で変位電流を無視し、

\[
\boldsymbol J=\sigma\boldsymbol E,\qquad
\nabla\times\boldsymbol H=\boldsymbol J,\qquad
\nabla\times\boldsymbol E=-\mu\frac{\partial\boldsymbol H}{\partial t}
\]

を組み合わせると、磁界および電流密度は拡散方程式

\[
\frac{\partial\boldsymbol H}{\partial t}
=
\frac{1}{\mu\sigma}\nabla^2\boldsymbol H,
\qquad
\frac{\partial\boldsymbol J}{\partial t}
=
\frac{1}{\mu\sigma}\nabla^2\boldsymbol J
\]

に従う。正弦波では表面からの深さ \(n\) に対し、おおよそ

\[
\boldsymbol J(n)
\propto
\exp\left[-(1+j)\frac{n}{\delta}\right],
\qquad
\delta
=
\sqrt{\frac{2}{\omega\mu\sigma}}
=
\sqrt{\frac{\rho}{\pi f\mu}}
\]

となる。`skin_depth_m()` がこの \(\delta\) を計算する。

導体を、電流密度の変化を解像できる厚さのフィラメントへ分割し、それぞれに抵抗と全相互部分インダクタンスを与えると、

\[
\boldsymbol Z\boldsymbol I
=
\left(
\boldsymbol R+j\omega\boldsymbol L^{\mathrm p}
\right)\boldsymbol I
\]

を満たすように回路自身が電流を分配する。外部導体や隣接フィラメントの磁界も
\(\boldsymbol L^{\mathrm p}\) の非対角項へ入るため、解像された方向については近接効果を別モデルとして追加する必要がない。

分割しても銅が物理的に切れるわけではない。各フィラメントを独立した sheet のままにすると、端子が最初に与えた電流配分を他層へ移せず、表皮効果を再現できない。
`filament_links()` は隣接フィラメントを全占有セルで抵抗性の縦枝により接続する。

理論上の十分条件は、電流を運ぶ各フィラメントについて
\(t_k\lesssim\delta/2\) とすることである。現行の `graded_filaments()` は表面を既定で
\(\delta/4\) に切る一方、電流が小さい中央部は \(\delta\) より厚くする。これは未知数を節約する近似であり、「全フィラメントが skin depth より薄い」という厳密な離散化ではない。中央部の近似誤差を含む測定済みの離散化天井は §3.4 を正本とする。

### 6.8 「2.5D」が成立する条件と周波数限界

ここでいう 2.5D は、面内では二次元格子を用い、厚さ方向は少数の既知の高さに置いたフィラメントとして表すことを意味する。完全な三次元電流分布を仮定しているわけではない。

| 条件 | 35 µm 箔 | 3.5 mm インレイ | 限界の意味 |
|---|---:|---:|---|
| 一つのフィラメント内で厚さ方向に一様、判定 \(t\le\delta/2\) | 約 **0.89 MHz** まで | 約 **89 Hz** まで | これを超えたら分割が必要 |
| 導体を少数の既知高さへ置ける | 銅厚による固有の破綻周波数なし | 同左 | 傾斜導体、連続的な高さ変化、未解像の側壁電流が現れれば幾何学的に破綻 |
| 遅延を無視した二次元 Green 核が使える | 銅厚ではなく基板最大寸法と誘電率で決まる | 同左 | 次節の準静的上限を適用 |

上の 0.89 MHz と 89 Hz は、`uniform_through_thickness()` の既定基準
\(t/\delta=0.5\) による工学的な境界であり、物理量が不連続に変化する周波数ではない。
300 kHz では 35 µm 箔は一フィラメントで扱えるが、3.5 mm インレイは扱えず、広幅平板の測定値では \(R_{\mathrm{ac}}/R_{\mathrm{dc}}=14.50\) である。銅の skin depth の測定値とこの比の正本は `docs/SHEET_PEEC.md` および
`tests/test_skin_filaments.py` に置く。

### 6.9 FR-4 をモデル化しないことの意味

FR-4 を \(\mu_r=1\) の非磁性体とみなす限り、磁気準静的なベクトルポテンシャルに磁気境界は生じない。従って磁気部分には

\[
G_m(\boldsymbol r,\boldsymbol r')
=
\frac{\mu_0}{4\pi\lVert\boldsymbol r-\boldsymbol r'\rVert}
\]

という自由空間 Green 関数をそのまま使える。この意味で、FR-4 を磁気演算子へ入れないことは近似ではない。ただし、これは \(\mu_r=1\) と遅延無視を前提にした意味での「正確」である。

一方、FR-4 の \(\epsilon_r\) は係数電位、導体間容量、変位電流へ入る。本ソルバはそれらを持たないため、

\[
\boldsymbol J_{\mathrm d}
=
j\omega\epsilon\boldsymbol E
\]

による power–ground 間電流や plane resonance を表現しない。無視が安全かどうかは周波数だけでは決まらず、plane-pair capacitance \(C_{\mathrm{pp}}\)、plane 間電圧
\(V_{\mathrm{pp}}\)、端子電流 \(I_{\mathrm{port}}\) に対して

\[
\eta_C
=
\frac{\omega C_{\mathrm{pp}}|V_{\mathrm{pp}}|}
{|I_{\mathrm{port}}|}
\ll 1
\]

であることを case ごとに確認しなければならない。等価的には、比較対象の return-path impedance を \(Z_{\mathrm{return}}\) として
\(\omega C_{\mathrm{pp}}|Z_{\mathrm{return}}|\ll1\) が必要である。従って、一般の PDN に対して「何 MHz までは常に安全」という単一の周波数は定義できない。

寸法だけから置ける別の上限は伝搬遅延である。最大寸法 \(L\) に対し
\(L<\lambda_g/10\) を準静的条件とすれば、

\[
f_{\mathrm{MQS,max}}
\approx
\frac{c}{10L\sqrt{\epsilon_r}}.
\]

34.7 mm 角、\(\epsilon_r\simeq4\) と仮定した場合は約 **0.43 GHz** であり、これを超える領域では容量の大小にかかわらず本モデルを安全とは扱わない。最初の半波共振の寸法値は約 2.2 GHz だが、\(\lambda/10\) の時点で既に準静的モデルの適用保証を外すべきである。逆に 0.43 GHz 未満でも、近接した power/ground plane や外付け回路によって上の \(\eta_C\) 条件を満たさなければ、容量省略は安全でない。現対象の 20 kHz／300 kHz では十分低いが、この判断は容量を実際に計算した検証結果ではなく、現行モデルの適用条件である。

[^ruehli]: A. E. Ruehli, [“Equivalent Circuit Models for Three-Dimensional Multiconductor Systems,”](https://doi.org/10.1109/TMTT.1974.1128204) IEEE Transactions on Microwave Theory and Techniques, 1974.
[^hoer-love]: C. Hoer and C. Love, [“Exact Inductance Equations for Rectangular Conductors with Applications to More Complicated Geometries,”](https://doi.org/10.6028/jres.069C.016) Journal of Research of the National Bureau of Standards, 1965.

---

## 7. 未解決課題の整理

優先度は、**P0** を正しさまたは standalone product としての出荷を妨げるもの、
**P1** を対象範囲または性能を大きく制約するもの、**P2** を現在の対象条件では影響が限定される拡張とする。

| 課題 | 欠けているもの | コストと現れる場所 | 完了条件 | 優先度と理由 |
|---|---|---|---|---|
| ~~縦枝間の部分インダクタンス~~ **解決** | — | 指摘どおり自己項のみ、しかも既定値 0 だった。3.5mm インレイのフィラメント界面で隣接列の相互項は自己項の 46〜69%、表面リンクの ωL/R は 0.46 で、無視できる量ではなかった | 面内と同じ畳み込みとして自己・相互を一体で実装した。「level」は縦枝が繋ぐ層対一つで、同一 level の枝は同じ高さを張るので結合は面内 offset のみに依存する。11 フィラメントで 10 level・55 表・1.1 MiB・構築 2.1 秒。level は stackup から推論せず明示させ、演算子が知らない level を持つ mesh は拒否する。回帰試験は `tests/test_sheet_peec.py` の `VerticalOperatorTests` | 解決済み。根拠は物理側にある（平行な枝は結合する、隣接列の相互項は自己項の 46〜69%、逆向きの再配分電流で相互項が自己項を打ち消す）。当初「射影の天井を超える値は収束解が取り得ない」と述べたが、これは誤りだった。天井は真の分布を mesh 上で表現したときの損失であり、ソルバは真の分布を再現する義務がないので、天井はソルバ出力の上界ではない。§5.2 参照 |
| 側壁の電流集中を解像できない | 面内 pitch より細い edge/side-wall basis | 300 kHz の \(\delta=0.1207\) mm に対して optimizer の pitch は 0.2 mm、すなわち \(1.66\delta\)。既定の厚さ分割だけでは、与えられた厳密分布が持つ損失の 77.2% しか保持できない。bulk P99 はまさにこの局所集中を読む | 最も単純には面内格子を全体で 0.1 mm 以下へ細分化し、FFT の並進不変性を保つ。局所 refinement を使うなら、FFT 遠方場と非一様な近傍補正を結ぶ multilevel/domain-decomposition 演算子が必要 | **P0**。現行 0.2 mm の厚銅 P99 と交流損失には、反復収束では回復できない既知の天井がある |
| optimizer 向け結果 adapter | `SheetSolution.node_voltage` と `branch_current` から、`SolveResult.current_density` および `bulk_p99_current_density_a_per_mm2` へ変換する経路 | `peec_fastopt/sheet_peec.py` は配列を返すだけである。一方 optimizer は `/home/hyamada/plane_opt_refactor/src/plane_opt/physics/resistive_mesh.py` の `SolveResult` と、`pypeec_solution_to_result()` が作る cell-wise density を読む。adapter がないため gate に接続できない | §1・§2 の定義に従い、枝中心電流から層・セルごとの複素電流密度へ写像し、端子セル除外、フィラメント集約、縦電流の別集計を実装する。既存二 backend と同じ fixture で契約試験を行う | **P0**。ソルバが解けても主要 consumer が結果を評価できない |
| end-to-end sheet 求解の境界条件と絶対検証 | 発達した表皮電流分布へ到達することを示す端子条件と測定手順 | 現在の bar 励振は端面電流を DC 比率で固定するため、最大横寸法/\(\pi\) 程度の entry length で交流分布へ緩和する必要がある。測定窓がその影響を除けていない。また `Terminal.current_a` は `float` なので、フィラメントごとに異なる位相を注入できない | 十分長い bar、等電位端面、または場の解の複素分布を与える端子を実装する。観測量は断面平均電位ではなく \(\sum_b R_b|I_b|^2/|I|^2\) とし、面内・厚さ方向を細分化して参照解へ収束することを示す | **P0**。現状は演算子が発達解を表現できることまでで、求解器が自身の離散化天井へ達することを示していない |
| ~~複数連結成分のゲージと電流収支~~ **解決** | — | 指摘どおり実在した。2成分の導体で系が厳密に特異になり、`MatrixRankWarning` を出して NaN を返していた。`solve_sheet_case()` の docstring が主張していた「上流の端子検査が1成分であることを確立している」という検査は存在しなかった | `_components()` と `_components_with_terminals()` を追加し、端子を持つ成分ごとに1節点を接地、成分ごとに電流収支を検査、端子を持たない島は `undriven_nodes` として報告して除外する。回帰試験は `tests/test_sheet_peec.py` の `ConnectedComponentTests` | 解決済み |
| PyPEEC の operator 再利用と case batching | operator preparation と sweep solve の間の再利用可能な公開境界 | optimizer の `solve_pypeec_multiterminal_case()` は case ごとに `run_mesher_data()` と `run_solver_data()` を呼ぶ。CUDA wrapper は `cuda_pypeec.py` の `_VOXEL_CACHE` で mesher 出力だけを再利用するが、solver operator は呼出しごとに準備される。power module の9 case、BLDC board の15 caseは、いずれも端子 pad 集合が全て異なるため、一つの既存 sweep にまとめる対象もない | PyPEEC 側に「geometry/operator prepare」と「source/frequency solve」の境界を設けるか、`peec_fastopt` が独自演算子を所有する。異なる pad partition を一つの prepared domain で扱える source API も必要 | **P1**。正しさではなく反復最適化の準備時間を支配する。なお PyPEEC 5.8 本体は本環境に未インストールなので、「公開 seam が存在しない」という upstream API の事実はローカル実行では再確認できず、現 adapter と既存設計文書から確認できる範囲に限る |
| sheet 経路の CUDA 実装 | kernel spectrum、2D FFT、saddle-point Krylov、前処理の device 実装 | `SheetInductanceOperator` は `numpy.fft.rfft2()`／`irfft2()` を使用し、`solve_sheet_case()` は SciPy GMRES/SuperLU を使用する。`cuda_pypeec.py` には別系統の PyPEEC/CuPy adapter があるが、sheet solver の CUDA 化ではなく、実機検証も別問題である | CuPy または backend-neutral array API へ演算子を移し、層対を batched cuFFT で処理する。incidence と Krylov を device resident にし、CPU 経路との複素解・残差一致を確認する | **P1**。standalone product の性能要求に直結するが、物理式の正しさは変えない |
| barrel の幾何からの自己・相互項と表皮効果 | drill 径、plating 厚、barrel 長からの部分インダクタンス生成と、barrel wall 内の交流電流分布 | `via_resistance()` は環状断面から DC 抵抗だけを計算する。`ViaBranch.inductance_h` は呼出側が与える scalar で、幾何から計算されない。barrel wall の厚さ方向・周方向の skin/proximity effect もない | 円筒殻または等価直方体による自己項を実装し、§7.1 の \(L_{zz}\) 相互項へ統合する。plating が \(\delta/2\) を超える場合は壁厚方向の filament または surface-impedance model を選択する | **P1**。通常の 25 µm plating は 300 kHz の \(\delta/2\) より薄いため現対象では wall skin effect は小さいが、filled via・厚 plating・高周波には一般化できない |
| graded filament の一般誤差保証 | 任意断面・任意周波数に対する a priori 誤差推定 | `graded_filaments()` は中央フィラメントを \(\delta\) より厚くできる。`discretisation_note()` の `ceiling_fraction` は一つの 1.6×3.5 mm bar、300 kHz の最近傍測定であり、保証境界ではない | 複数の aspect ratio、周波数、端部条件に対する場の解を作り、面内 pitch と filament grading の二変数収束モデルを構築する。許容誤差を満たさない入力は求解前に拒否する | **P1**。現在の測定対象には根拠があるが、standalone product の一般入力には外挿できない |
| 容量・誘電体・遅延 | 係数電位、電荷未知数、誘電体境界、retarded Green 関数 | power/ground plane 間の変位電流、plane resonance、容量性 return path を表現できない。§6.9 の \(\eta_C\) が小さくない case、または約 0.43 GHz の寸法上限を超える case では適用外になる | quasi-static capacitive PEEC として係数電位行列・電荷保存を追加し、必要なら layered-medium／retarded Green 関数へ拡張する。少なくとも入力から \(\eta_C\) と電気長を評価して適用外を拒否する | **P1**。20 kHz／300 kHz の現対象では低いが、周波数上限を持つ standalone field solver には不可欠 |

### 7.1 調査の結果、未解決課題から除外する項目

「異なる銅厚の層が `SheetInductanceOperator` に拒否される」という記述は、現行コードには当てはまらない。

`SheetInductanceOperator.__init__()` は層対の両側について別々の `CellGeometry` を作り、二層目を `build_kernel(..., other=other)` として渡している。
`build_kernel()` が一致を要求するのは `length_m` と `width_m`、すなわち面内格子寸法だけであり、`thickness_m` は異なってよい。これは graded filament の異厚層対を扱うための現行仕様である。

従って理論上の欠落としては除外する。ただし、異厚の閉形式に対する reciprocity test は
`tests/test_sheet_inductance.py` にある一方、異厚 stackup を
`SheetInductanceOperator` から `solve_sheet_case()` まで通す明示的な回帰試験は見当たらない。追加すべきものは「異厚対応の実装」ではなく、その end-to-end 回帰試験である。

指摘された end-to-end 回帰試験は追加した（`tests/test_sheet_peec.py` の
`MixedThicknessTests`）。異厚の2層を両端でビア接続し、面内抵抗の比に従って分流する
ことを確認する。最初に書いた試験は中央のビア1本で分流を測ろうとしており、上流の層は
分流できないので想定が誤っていた。

