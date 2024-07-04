#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from alifebook_lib.visualizers import SwarmVisualizer
from vispy.scene import visuals
import vispy.app  # vispy.app モジュールをインポートする

# visualizerの初期化 (Appendix参照)
visualizer = SwarmVisualizer()

# シミュレーションパラメタ
N = 256
# 力の強さ
COHESION_FORCE = 0.008
SEPARATION_FORCE = 0.4
ALIGNMENT_FORCE = 0.06
# 力の働く距離
COHESION_DISTANCE = 0.5
SEPARATION_DISTANCE = 0.05
ALIGNMENT_DISTANCE = 0.1
# 力の働く角度
COHESION_ANGLE = np.pi / 2
SEPARATION_ANGLE = np.pi / 2
ALIGNMENT_ANGLE = np.pi / 3
# 速度の上限/下限
MIN_VEL = 0.005
MAX_VEL = 0.03
# 境界で働く力（0にすると自由境界）
BOUNDARY_FORCE = 0.001
# エサに吸引される力と動かす間隔
PREY_FORCE = 0.0005
PREY_MOVEMENT_STEP = 500
# 敵から避ける力と敵の移動間隔
PREDATOR_FORCE = 0.2
PREDATOR_DISTANCE = 0.5
PREDATOR_MOVEMENT_STEP = 500
PREDATOR_TARGET_UPDATE_STEP = 1000  # プレデターのターゲット更新周期
PREDATOR_SPEED = 0.01  # プレデターの速度（例）

# 位置と速度
x = np.random.rand(N, 3) * 2 - 1
v = (np.random.rand(N, 3) * 2 - 1) * MIN_VEL
# エサの位置
prey_x = np.random.rand(1, 3) * 2 - 1
# 敵の位置
predator_x = np.random.rand(1, 3) * 2 - 1

# cohesion, separation, alignmentの３つの力を代入する変数
dv_coh = np.empty((N, 3))
dv_sep = np.empty((N, 3))
dv_ali = np.empty((N, 3))
# 境界で働く力を代入する変数
dv_boundary = np.empty((N, 3))
# 敵から避ける力を代入する変数
dv_predator = np.empty((N, 3))

# プレデターが追うターゲットのインデックス
predator_target_idx = None

# 青色のマーカーを表示するためのマーカーオブジェクトを作成
def create_marker_visuals(visualizer, positions, color=(0, 0, 1), size=20):
    if not hasattr(visualizer, '_blue_markers'):
        visualizer._blue_markers = visuals.Markers(parent=visualizer._view.scene)
    visualizer._blue_markers.set_data(positions, face_color=color, size=size)
    visualizer._canvas.update()
    vispy.app.process_events()

t = 0
while visualizer:
    for i in range(N):
        # ここで計算する個体の位置と速度
        x_this = x[i]
        v_this = v[i]
        # それ以外の個体の位置と速度の配列
        x_that = np.delete(x, i, axis=0)
        v_that = np.delete(v, i, axis=0)
        # 個体間の距離と角度
        distance = np.linalg.norm(x_that - x_this, axis=1)
        angle = np.arccos(np.dot(v_this, (x_that - x_this).T) / (np.linalg.norm(v_this) * np.linalg.norm((x_that - x_this), axis=1)))
        # 各力が働く範囲内の個体のリスト
        coh_agents_x = x_that[(distance < COHESION_DISTANCE) & (angle < COHESION_ANGLE)]
        sep_agents_x = x_that[(distance < SEPARATION_DISTANCE) & (angle < SEPARATION_ANGLE)]
        ali_agents_v = v_that[(distance < ALIGNMENT_DISTANCE) & (angle < ALIGNMENT_ANGLE)]
        # 各力の計算
        dv_coh[i] = COHESION_FORCE * (np.average(coh_agents_x, axis=0) - x_this) if (len(coh_agents_x) > 0) else 0
        dv_sep[i] = SEPARATION_FORCE * np.sum(x_this - sep_agents_x, axis=0) if (len(sep_agents_x) > 0) else 0
        dv_ali[i] = ALIGNMENT_FORCE * (np.average(ali_agents_v, axis=0) - v_this) if (len(ali_agents_v) > 0) else 0
        dist_center = np.linalg.norm(x_this)  # 原点からの距離
        dv_boundary[i] = - BOUNDARY_FORCE * x_this * (dist_center - 1) / dist_center if (dist_center > 1) else 0

        # プレデターから避ける力の計算
        dist_predator = np.linalg.norm(predator_x - x_this)
        if dist_predator < PREDATOR_DISTANCE:
            dv_predator[i] = PREDATOR_FORCE * (x_this - predator_x) / dist_predator
        else:
            dv_predator[i] = 0

    # 速度のアップデートと上限/下限のチェック
    v += dv_coh + dv_sep + dv_ali + dv_boundary + dv_predator
    # エサへの吸引力を加える
    v += PREY_FORCE * (prey_x - x) / np.linalg.norm((prey_x - x), axis=1, keepdims=True) ** 2

    if t % PREY_MOVEMENT_STEP == 0:
        prey_x = np.random.rand(1, 3) * 2 - 1

    if t % PREDATOR_MOVEMENT_STEP == 0:
        predator_x = np.random.rand(1, 3) * 2 - 1

    # プレデターのターゲット選択
    if t % PREDATOR_TARGET_UPDATE_STEP == 0:
        distances_to_predator = np.linalg.norm(x - predator_x, axis=1)
        predator_target_idx = np.argmin(distances_to_predator)

    # プレデターのターゲットに基づいた動き
    if predator_target_idx is not None:
        target_position = x[predator_target_idx]
        direction_to_target = target_position - predator_x
        direction_to_target_norm = np.linalg.norm(direction_to_target)
        if direction_to_target_norm > 0:
            predator_v = PREDATOR_SPEED * direction_to_target / direction_to_target_norm
        else:
            predator_v = np.zeros_like(predator_x)
        predator_x += predator_v

    # エサと敵の位置を表示
    create_marker_visuals(visualizer, np.vstack(predator_x))
    visualizer.set_markers(np.vstack(prey_x))
    
    t += 1

    for i in range(N):
        v_abs = np.linalg.norm(v[i])
        if v_abs < MIN_VEL:
            v[i] = MIN_VEL * v[i] / v_abs
        elif v_abs > MAX_VEL:
            v[i] = MAX_VEL * v[i] / v_abs

    # 位置のアップデート
    x += v
    visualizer.update(x, v)
