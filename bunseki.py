import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 物理定数とWASP-39bのパラメータ ---
# (変更なし)
G = 6.67430e-11
R_JUP = 6.9911e7
M_JUP = 1.898e27
R_SUN = 6.957e8
R_planet_base = 1.27 * R_JUP
M_planet = 0.28 * M_JUP
R_star = 0.90 * R_SUN
T_eq = 1150
g = G * M_planet / (R_planet_base**2)
mu = 2.3 * 1.66e-27
k_B = 1.38e-23
H = (k_B * T_eq) / (mu * g)

print(f"スケールハイト: {H / 1000:.2f} km")

# --- 2. フィッティング・パラメータ ---
# VMRとC_scalingを調整してフィッティングを行います
P_ref = 100.0  # 基準圧力 (Pa)
C_scaling = 5e13  # ★ 全体の吸収強度を調整する新しいスケーリング定数
vmr_values = {
    'C2H2': 5e-12, 'CH4': 2e-10, 'CO': 1e-9, 'CO2': 2e-10,
    'H2O': 8e-9, 'H2S': 2e-8, 'HCN': 5e-12, 'NH3': 5e-9,
    'SiO': 5e-10, 'SO2': 5e-10,
}

molecule_files = {name: f'xs_{name}_g395h.csv' for name in vmr_values.keys()}
n_total = P_ref / (k_B * T_eq)

# --- 3. データの読み込みとモデル計算 ---
try:
    col_names = ['wavelength_nm', 'radius_ratio', 'err_p', 'err_n']
    spec_df = pd.read_csv('spectrum.csv', comment='#', names=col_names)
    spec_df['observed_depth'] = spec_df['radius_ratio']**2
    spec_df['depth_err'] = 2 * spec_df['radius_ratio'] * (spec_df['err_p'] + spec_df['err_n']) / 2
except FileNotFoundError:
    print("エラー: spectrum.csv が見つかりません。")
    exit()

min_depth_ratio = spec_df['radius_ratio'].min()
R_p_ref = min_depth_ratio * R_star
base_depth = min_depth_ratio**2

# ★ 個々の分子の寄与と、全体の寄与を計算
individual_delta_depths = {}
total_opacity_scaled = np.zeros(len(spec_df))

for molecule_name, file_path in molecule_files.items():
    vmr = vmr_values.get(molecule_name, 0)
    if vmr == 0:
        continue
    try:
        cross_section_df = pd.read_csv(file_path, header=None, names=['cross_section_cm2'])
        cross_section_m2 = cross_section_df['cross_section_cm2'] * 1e-4
        
        # この分子のスケールされた光学的厚みを計算
        opacity_molecule_scaled = n_total * vmr * C_scaling * cross_section_m2
        
        # 全体の光学的厚みに加算
        total_opacity_scaled += opacity_molecule_scaled
        
        # ★ この分子単独での寄与(delta_depth)も計算して保存しておく
        z_molecule = H * np.log(1 + opacity_molecule_scaled)
        delta_depth_molecule = (2 * R_p_ref * z_molecule + z_molecule**2) / R_star**2
        individual_delta_depths[molecule_name] = delta_depth_molecule
        
        print(f"'{molecule_name}' のモデルを計算しました。")
    except FileNotFoundError:
        print(f"警告: {file_path} が見つかりません。スキップします。")

# 混合大気モデルの最終計算
z_total = H * np.log(1 + total_opacity_scaled)
total_delta_depth = (2 * R_p_ref * z_total + z_total**2) / R_star**2
combined_model_depth = base_depth + total_delta_depth

# --- 4. グラフのプロット (★修正箇所) ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(15, 9))

# 観測データをプロット
ax.errorbar(
    spec_df['wavelength_nm'] / 1000, spec_df['observed_depth'] * 1e6,
    yerr=spec_df['depth_err'] * 1e6, fmt='o', capsize=1, color='black',
    markersize=2, label='Observed Data (WASP-39b)', alpha=0.6, zorder=1
)

# ★ 個々の分子の寄与をプロット
colors = plt.cm.tab10(np.linspace(0, 1, len(vmr_values)))
mol_index = 0
for molecule_name, delta_depth_mol in individual_delta_depths.items():
    model_depth_mol = base_depth + delta_depth_mol
    ax.plot(
        spec_df['wavelength_nm'] / 1000, model_depth_mol * 1e6,
        linewidth=1.5, linestyle='--',
        label=f'{molecule_name} (VMR={vmr_values.get(molecule_name, 0):.1e})',
        alpha=0.8, color=colors[mol_index % 10], zorder=2
    )
    mol_index += 1

# 混合大気モデル（最終モデル）をプロット
ax.plot(
    spec_df['wavelength_nm'] / 1000, combined_model_depth * 1e6,
    linewidth=3, label='Combined Model (All Molecules)', color='red', zorder=10
)


ax.set_xlabel('Wavelength (μm)', fontsize=14)
ax.set_ylabel('Transit Depth (ppm)', fontsize=14)
ax.set_title('Transmission Spectrum of WASP-39b with Individual Contributions', fontsize=16)
ax.legend(fontsize=10, loc='upper left',ncol=6)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('transmission_spectrum_WASP-39b_with_contributions.png', dpi=300)
plt.show()