# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from math_verify import (  # @manual=fbsource//third-party/pypi/math-verify:math-verify
    parse,
)


import json
import re


def delta1_metric_token_depth_map(contents, solution, **kwargs):
    """Reward function for token-level depth map formatted answers.
    
    Both content and solution are JSON arrays: [d0, d1, ..., d2699]
    """
    rewards = []
    for content, sol in zip(contents, solution):
        try:
            gt_depths = json.loads(sol.strip())

            # Parse model output (may be truncated)
            try:
                pred_depths = json.loads(content.strip())
            except json.JSONDecodeError:
                # Try to recover truncated JSON array
                text = content.strip()
                if text.startswith("[") and not text.endswith("]"):
                    trimmed = text.rstrip().rstrip(",")
                    try:
                        pred_depths = json.loads(trimmed + "]")
                    except json.JSONDecodeError:
                        last_comma = trimmed.rfind(",")
                        if last_comma > 0:
                            try:
                                pred_depths = json.loads(trimmed[:last_comma] + "]")
                            except json.JSONDecodeError:
                                pred_depths = None
                        else:
                            pred_depths = None
                else:
                    pred_depths = None

            if pred_depths is None or not isinstance(pred_depths, list):
                print(f"error: Failed to parse depth map output, first 200 chars: {content[:200]}")
                rewards.append(0.0)
                continue

            if len(pred_depths) < len(gt_depths):
                print(f"warning: Predicted {len(pred_depths)}/{len(gt_depths)} depth values")

            # Compute per-token delta1
            point_rewards = []
            for i, gt_val in enumerate(gt_depths):
                gt_val = float(gt_val)
                if gt_val <= 0:
                    continue
                if i < len(pred_depths):
                    try:
                        pred_val = float(pred_depths[i])
                        if pred_val <= 0:
                            point_rewards.append(0.0)
                        else:
                            point_rewards.append(
                                1.0 if max(pred_val / gt_val, gt_val / pred_val) < 1.25 else 0.0
                            )
                    except (ValueError, TypeError):
                        point_rewards.append(0.0)
                else:
                    point_rewards.append(0.0)

            rewards.append(sum(point_rewards) / len(point_rewards) if point_rewards else 0.0)
        except Exception as e:
            print(f"error: {e} during depth map parsing, content = {content[:200]}")
            rewards.append(0.0)

    return rewards


METRIC_CLASSES = {
    "delta1_metric_token_depth_map": delta1_metric_token_depth_map,
}
