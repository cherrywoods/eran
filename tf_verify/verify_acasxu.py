# This file contains an modified excerpt from __main__
# for verifying ACASXu and similar networks that are stored as ONNX files
# but accept an additional batch dimension as input (as opposed to what __main__ expects)
from typing import Tuple, Optional, List, Sequence

import os
from logging import info, warning
from tqdm import tqdm

# import sys
# Note: these two have to be added to PYTHONPATH (or be otherwise made accessible)
# sys.path.insert(0, '../ELINA/python_interface/')
# sys.path.insert(0, '../deepg/code/')
import numpy as np
import onnxruntime.backend as rt
from multiprocessing import Value, Pool

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
    # input = input.reshape(1, math.prod(input.shape))
    return base.run(input)


def _estimate_grads(specLB, specUB, model, dim_samples=3, input_shape=(1, )):
    # Estimate gradients using central difference quotient and average over dim_samples+1 in the range of the input bounds
    # Very computationally costly
    specLB = np.array(specLB, dtype=np.float32)
    specUB = np.array(specUB, dtype=np.float32)
    inputs = [(((dim_samples - i) * specLB + i * specUB) / dim_samples).reshape(*input_shape)
              for i in range(dim_samples + 1)]
    diffs = np.zeros(len(specLB))

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
        -> Tuple[bool, Optional[Sequence[np.ndarray]]]:
    hold, nn, nlb, nub, _, _ = \
        eran.analyze_box(specLB, specUB, domain, timeout_lp, timeout_milp, use_default_heuristic, constraints)
    if hold:
        return hold, []
    elif depth >= max_depth:
        if failed_already.value and complete:
            verified_flag, adv_examples, _ = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
            xs = []
            if not verified_flag:
                if adv_examples is not None:
                    for adv_image in adv_examples:
                        hold, _, nlb, nub, _, x = eran.analyze_box(adv_image, adv_image,
                                                                   domain, timeout_lp, timeout_milp,
                                                                   use_default_heuristic, constraints)
                        print(f'hold={hold}, adv_image={adv_image}')
                        if not hold:
                            info(f"property violated at {adv_image} output_score {nlb[-1]}")
                            failed_already.value = 0
                            xs.append(adv_image)
                            # break
            return verified_flag, xs if adv_examples is not None else None
        else:
            return False, None
    else:
        # grads = _estimate_grads(specLB, specUB, model, input_shape=eran.input_shape)
        # grads + small epsilon so if gradient estimation becomes 0 it will divide the biggest interval.
        # smears = np.multiply(grads + 0.00001, [u-l for u, l in zip(specUB, specLB)])

        nn.set_last_weights(constraints)
        grads_lower, grads_upper = nn.back_propagate_gradient(nlb, nub)
        smears = [max(-grad_l, grad_u) * (u-l)
                  for grad_l, grad_u, l, u in zip(grads_lower, grads_upper, specLB, specUB)]

        index = np.argmax(smears)
        m = (specLB[index]+specUB[index])/2

        result = failed_already.value
        xs = []
        if result:
            result1, xs1 = _acasxu_recursive(
                specLB, [ub if i != index else m for i, ub in enumerate(specUB)],
                model, eran, constraints, failed_already,
                max_depth, depth + 1,
                domain, timeout_lp, timeout_milp, use_default_heuristic, complete
            )
            result = result and result1
            if xs1 is not None:
                xs.extend(xs1)
        if result:
            result2, xs2 = _acasxu_recursive(
                [lb if i != index else m for i, lb in enumerate(specLB)], specUB,
                model, eran, constraints, failed_already,
                max_depth, depth + 1,
                domain, timeout_lp, timeout_milp, use_default_heuristic, complete
            )
            result = result and result2
            if xs2 is not None:
                xs.extend(xs2)
        return result, xs


def _start_acasxu_recursive(kwargs):
    """
    Utility method to start _acasxu_recursive through multiprocessing
    """
    network_file = kwargs['network_file']
    model, _ = read_onnx_net(network_file)
    eran = ERAN(model, is_onnx=True)
    global _failed_already

    return _acasxu_recursive(
        kwargs['specLB'], kwargs['specUB'], model, eran,
        kwargs['constraints'], _failed_already,
        kwargs['max_depth'], 0,
        kwargs['domain'], kwargs['timeout_lp'], kwargs['timeout_milp'],
        kwargs['use_default_heuristic'], kwargs['complete']
    )


def _init(failed_already):
    """
    Method to initialize a multiprocessing pool for running _acasxu_recursive
    """
    global _failed_already
    _failed_already = failed_already


def _init_domain(domain):
    if domain == 'refinezono':
        return 'deepzono'
    elif domain == 'refinepoly':
        return 'deeppoly'
    else:
        return domain


def verify_acasxu(network_file: str, means: np.ndarray, stds: np.ndarray,
                  input_boxes: List[List[Tuple[np.ndarray, np.ndarray]]],
                  output_constraints: List[List[Tuple[int, int, float]]],
                  timeout_lp=1, timeout_milp=1, max_depth=10,
                  use_default_heuristic=True, complete=True
                  ) -> Optional[List[np.ndarray]]:
    """
    Verifies an ACASXu network. Probably also works for other networks in other settings.

    Returns None if an no counterexamples could be found and the network could not be verified.
    Returns an empty list if the network could be verified.
    Returns a list of inputs for which the network violates the constraints if the network could not be verified.
    """
    domain = "deeppoly"

    model, is_conv = read_onnx_net(network_file)
    eran = ERAN(model, is_onnx=True)

    for box_index, box in enumerate(input_boxes):
        # total will be updated later
        progress_bar = tqdm(total=1)

        specLB = [interval[0] for interval in box]
        specUB = [interval[1] for interval in box]
        _normalize(specLB, means, stds)
        _normalize(specUB, means, stds)

        counterexample_list = []
        # adex_holds stores whether x_adex (below) is actually a counterexample
        # if adex_holds is True, then x_adex is a spurious counterexample
        adex_holds = True

        verified_flag, nn, nlb, nub, _, x_adex = eran.analyze_box(
            specLB, specUB, _init_domain(domain),
            timeout_lp, timeout_milp, use_default_heuristic, output_constraints
        )

        if not verified_flag and x_adex is not None:
            adex_holds, _, _, _, _, _ = eran.analyze_box(
                x_adex, x_adex, "deeppoly",
                timeout_lp, timeout_milp,
                use_default_heuristic, output_constraints
            )
            if not adex_holds:
                verified_flag = False
                # we need to undo the input normalisation, that was applied to the counterexamples
                counterexample_list.append(np.array(x_adex) * stds + means)

        if not verified_flag and adex_holds:
            # expensive min/max gradient calculation
            nn.set_last_weights(output_constraints)
            grads_lower, grads_upper = nn.back_propagate_gradient(nlb, nub)

            smears = [max(-grad_l, grad_u) * (u-l)
                      for grad_l, grad_u, l, u in zip(grads_lower, grads_upper, specLB, specUB)]
            split_multiple = 20 / np.sum(smears)

            num_splits = [int(np.ceil(smear * split_multiple)) for smear in smears]
            step_size = []
            for i in range(len(specLB)):  # for each input dimension
                if num_splits[i] == 0:
                    num_splits[i] = 1
                step_size.append((specUB[i]-specLB[i])/num_splits[i])

            start_val = np.copy(specLB)
            end_val = np.copy(specUB)
            # _, nn, _, _, _, _ = eran.analyze_box(
            #     specLB, specUB, domain,
            #     timeout_lp, timeout_milp, use_default_heuristic, output_constraints
            # )

            # generate all combinations of splits of the input dimensions
            multi_bounds = [(specLB.copy(), specUB.copy())]
            for d in range(len(specLB)):  # for each input dimension
                # for each split from the previous dimensions
                new_multi_bounds = []
                for specLB_, specUB_ in multi_bounds:
                    for i in range(num_splits[d]):
                        specLB_ = specLB_.copy()
                        specUB_ = specUB_.copy()
                        specLB_[d] = start_val[d] + i*step_size[d]
                        specUB_[d] = np.fmin(end_val[d], start_val[d] + (i+1)*step_size[d])
                        new_multi_bounds.append((specLB_, specUB_))
                multi_bounds = new_multi_bounds

            # print(f"len(multi_bounds)={len(multi_bounds)}\n"
            #       f"means={means}\n"
            #       f"stds={stds}\n"
            #       f"grads_lower={grads_lower}\n"
            #       f"grads_upper={grads_upper}\n"
            #       f"smeans={smears}\n"
            #       f"num_splits={num_splits}\n"
            #       f"step_size={step_size}\n"
            #       f"start_val={start_val}\n"
            #       f"end_val={end_val}")

            progress_bar.reset(total=len(multi_bounds) + 1)
            progress_bar.update()  # for the first analyze_box run

            failed_already = Value('i', 1)
            pool = None
            try:
                # sequential version
                # res = itertools.starmap(
                #    lambda lb, ub: _acasxu_recursive(lb, ub, model, eran, output_constraints, failed_already,
                #                                     25, 0, domain, timeout_lp, timeout_milp, use_default_heuristic,
                #                                     complete),
                #    multi_bounds
                # )
                arguments = [
                    {
                        'specLB': lb, 'specUB': ub, 'network_file': network_file, 'constraints': output_constraints,
                        'max_depth': max_depth, 'domain': domain, 'timeout_lp': timeout_lp,
                        'timeout_milp': timeout_milp, 'use_default_heuristic': use_default_heuristic,
                        'complete': complete
                    }
                    for lb, ub in multi_bounds
                ]
                # using only half of the CPUs sped up the computation on the computers it was tested on
                # and also kept CPU utilisation high for all CPUs.
                pool = Pool(processes=os.cpu_count() // 2, initializer=_init, initargs=(failed_already,))
                res = pool.imap_unordered(_start_acasxu_recursive, arguments)

                counterexample_list = []
                verified_flag = True
                for verified, counterexamples in res:
                    if not verified:
                        verified_flag = False
                        if counterexamples is not None:
                            # convert counterexamples to numpy
                            counterexamples = [np.array(cx) for cx in counterexamples]
                            # we need to undo the input normalisation, that was applied to the counterexamples
                            counterexamples = [cx * stds + means for cx in counterexamples]
                            counterexample_list.extend(counterexamples)
                        else:
                            warning(f"ACASXu property not verified for Box {box_index+1} out of {len(input_boxes)} "
                                    f"without counterexample")
                    progress_bar.update()

            except Exception as ex:
                warning(f"ACASXu property not verified for Box {box_index+1} out of {len(input_boxes)} "
                        f"because of an exception: {ex}")
                raise ex
            finally:
                progress_bar.close()
                if pool is not None:
                    # make sure the Pool is properly closed
                    pool.terminate()
                    pool.join()
        else:
            # property has been verified in first analyze_box run.
            progress_bar.update()
            progress_bar.close()

        if not verified_flag and len(counterexample_list) > 0:
            info(f"ACASXu property not verified for Box {box_index + 1} out of {len(input_boxes)} "
                 f"with counterexamples: {counterexample_list}")
            return counterexample_list
        elif not verified_flag:
            info(f"ACASXu property not verified for Box {box_index + 1} out of {len(input_boxes)} "
                 f"without counterexamples")
            # raise RuntimeError("Property disproven, but no counterexample found.")
            return None
        else:
            info(f"ACASXu property verified for Box {box_index + 1} out of {len(input_boxes)}")
            return []
