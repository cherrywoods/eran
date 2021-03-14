# This file contains an modified excerpt from __main__
# for verifying ACASXu and similar networks that are stored as ONNX files
# but accept an additional batch dimension as input
from typing import Tuple, Optional, List

from logging import info, warning

# import sys
# Note: these two have to be added to PYTHONPATH (or be otherwise made accessible)
# sys.path.insert(0, '../ELINA/python_interface/')
# sys.path.insert(0, '../deepg/code/')
import numpy as np
import onnxruntime.backend as rt
import itertools
from multiprocessing import Value

from ai_milp import verify_network_with_milp
from read_net_file import read_onnx_net
from eran import ERAN


def _normalize(image, means, stds):
    # normalization taken out of the network
    if len(means) == len(image):
        for i in range(len(image)):
            image[i] -= means[i]
            if stds is not None:
                image[i] /= stds[i]


def _normalize_plane(plane, mean, std, channel, is_constant):
    plane_ = plane.clone()

    if is_constant:
        plane_ -= mean[channel]

    plane_ /= std[channel]

    return plane_


def _normalize_poly(num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights, uexpr_dim, means, stds):
    # normalization taken out of the network
    for i in range(len(lexpr_cst)):
        lexpr_cst[i] = (lexpr_cst[i] - means[i % 3]) / stds[i % 3]
        uexpr_cst[i] = (uexpr_cst[i] - means[i % 3]) / stds[i % 3]
    for i in range(len(lexpr_weights)):
        lexpr_weights[i] /= stds[(i // num_params) % 3]
        uexpr_weights[i] /= stds[(i // num_params) % 3]


def _onnx_predict(base, input):
    # add additional batch dimension
    input = input.reshape(1, len(input))
    return base.run(input)


def _estimate_grads(specLB, specUB, model, dim_samples=3):
    specLB = np.array(specLB, dtype=np.float32)
    specUB = np.array(specUB, dtype=np.float32)
    inputs = [((dim_samples - i) * specLB + i * specUB) / dim_samples for i in range(dim_samples + 1)]
    diffs = np.zeros(len(specLB))

    # refactor this out of this method
    # ONNX assumed
    runnable = rt.prepare(model, 'CPU')

    for sample in range(dim_samples + 1):
        pred = _onnx_predict(runnable, inputs[sample])

        for index in range(len(specLB)):
            if sample < dim_samples:
                l_input = [m if i != index else u for i, m, u in
                           zip(range(len(specLB)), inputs[sample], inputs[sample + 1])]
                l_input = np.array(l_input, dtype=np.float32)
                l_i_pred = _onnx_predict(runnable, l_input)
            else:
                l_i_pred = pred
            if sample > 0:
                u_input = [m if i != index else l for i, m, l in
                           zip(range(len(specLB)), inputs[sample], inputs[sample - 1])]
                u_input = np.array(u_input, dtype=np.float32)
                u_i_pred = _onnx_predict(runnable, u_input)
            else:
                u_i_pred = pred
            diff = np.sum([abs(li - m) + abs(ui - m) for li, m, ui in zip(l_i_pred, pred, u_i_pred)])
            diffs[index] += diff
    return diffs / dim_samples


def _acasxu_recursive(specLB, specUB, model, eran: ERAN, constraints, failed_already, max_depth=10, depth=0,
                      domain="deeppoly", timeout_lp=1, timeout_milp=1, use_default_heuristic=True,
                      complete=True) \
        -> Tuple[bool, Optional[np.ndarray]]:
    hold, nn, nlb, nub, _, x = \
        eran.analyze_box(specLB, specUB, domain, timeout_lp, timeout_milp, use_default_heuristic, constraints)
    if hold:
        return hold, None
    elif depth >= max_depth:
        if failed_already.value and complete:
            verified_flag, adv_examples = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
            x = None
            if not verified_flag:
                if adv_examples is not None:
                    for adv_image in adv_examples:
                        hold, _, nlb, nub, _, x = eran.analyze_box(adv_image, adv_image,
                                                                   domain, timeout_lp, timeout_milp,
                                                                   use_default_heuristic, constraints)
                        if not hold:
                            info("property violated at ", adv_image, "output_score", nlb[-1])
                            failed_already.value = 0
                            break
            return verified_flag, x
        else:
            return False, None
    else:
        grads = _estimate_grads(specLB, specUB, model)
        # grads + small epsilon so if gradient estimation becomes 0 it will divide the biggest interval.
        smears = np.multiply(grads + 0.00001, [u-l for u, l in zip(specUB, specLB)])

        index = np.argmax(smears)
        m = (specLB[index]+specUB[index])/2

        result = failed_already.value and _acasxu_recursive(
            specLB, [ub if i != index else m for i, ub in enumerate(specUB)],
            model, eran, constraints, failed_already,
            max_depth, depth + 1,
            domain, timeout_lp, timeout_milp, use_default_heuristic, complete
        )
        result = failed_already.value and result and _acasxu_recursive(
            [lb if i != index else m for i, lb in enumerate(specLB)], specUB,
            model, eran, constraints, failed_already,
            max_depth, depth + 1,
            domain, timeout_lp, timeout_milp, use_default_heuristic, complete
        )
        return result


def verify_acasxu(network_file: str, means: np.ndarray, stds: np.ndarray,
                  input_boxes: List[List[Tuple[np.ndarray, np.ndarray]]],
                  output_constraints: List[List[Tuple[int, int, float]]],
                  timeout_lp=1, timeout_milp=1, use_default_heuristic=True, complete=True
                  ) -> Optional[np.ndarray]:
    """
    Verifies an ACASXu network. Probably also works for other networks in other settings.
    """
    domain = "deeppoly"

    model, is_conv = read_onnx_net(network_file)
    eran = ERAN(model, is_onnx=True)

    for box_index, box in enumerate(input_boxes):
        specLB = [interval[0] for interval in box]
        specUB = [interval[1] for interval in box]
        _normalize(specLB, means, stds)
        _normalize(specUB, means, stds)

        _, nn, nlb, nub, _, _ = eran.analyze_box(
            specLB, specUB, domain,
            timeout_lp, timeout_milp, use_default_heuristic, output_constraints
        )
        # expensive min/max gradient calculation
        nn.set_last_weights(output_constraints)
        grads_lower, grads_upper = nn.back_propagate_gradiant(nlb, nub)

        smears = [max(-grad_l, grad_u) * (u-l) for grad_l, grad_u, l, u in zip(grads_lower, grads_upper, specLB, specUB)]
        split_multiple = 20 / np.sum(smears)

        num_splits = [int(np.ceil(smear * split_multiple)) for smear in smears]
        step_size = []
        for i in range(5):
            if num_splits[i] == 0:
                num_splits[i] = 1
            step_size.append((specUB[i]-specLB[i])/num_splits[i])

        start_val = np.copy(specLB)
        end_val = np.copy(specUB)
        _, nn, _, _, _, _ = eran.analyze_box(
            specLB, specUB, domain,
            timeout_lp, timeout_milp, use_default_heuristic, output_constraints
        )
        multi_bounds = []

        for i in range(num_splits[0]):
            specLB[0] = start_val[0] + i*step_size[0]
            specUB[0] = np.fmin(end_val[0], start_val[0] + (i+1)*step_size[0])

            for j in range(num_splits[1]):
                specLB[1] = start_val[1] + j*step_size[1]
                specUB[1] = np.fmin(end_val[1], start_val[1] + (j+1)*step_size[1])

                for k in range(num_splits[2]):
                    specLB[2] = start_val[2] + k*step_size[2]
                    specUB[2] = np.fmin(end_val[2], start_val[2] + (k+1)*step_size[2])
                    for l in range(num_splits[3]):
                        specLB[3] = start_val[3] + l*step_size[3]
                        specUB[3] = np.fmin(end_val[3], start_val[3] + (l+1)*step_size[3])
                        for m in range(num_splits[4]):

                            specLB[4] = start_val[4] + m*step_size[4]
                            specUB[4] = np.fmin(end_val[4], start_val[4] + (m+1)*step_size[4])

                            # add bounds to input for multiprocessing map
                            multi_bounds.append((specLB.copy(), specUB.copy()))

        failed_already = Value('i', 1)
        try:
            res = itertools.starmap(
                lambda lb, ub: _acasxu_recursive(lb, ub, model, eran, output_constraints, failed_already,
                                                 max_depth=10, depth=0, domain=domain, timeout_lp=timeout_lp,
                                                 timeout_milp=timeout_milp, use_default_heuristic=use_default_heuristic,
                                                 complete=complete),
                multi_bounds
            )

            if all(verified for verified, _ in res):
                info(f"ACASXu property verified for Box {box_index} out of {len(input_boxes)}")
                return None
            else:
                info(f"ACASXu property failed for Box {box_index} out of {len(input_boxes)}")
                for _, counterexample in res:
                    if counterexample is not None:
                        return counterexample
        except Exception as e:
            warning(f"ACASXu property failed for Box {box_index} out of {len(input_boxes)} because of an exception {e}")
            raise e
