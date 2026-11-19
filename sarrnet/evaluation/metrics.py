"""S-measure, E-measure, F-measure and MAE.

Adapted from PySODMetrics (https://github.com/lartpang/PySODMetrics, MIT License).
Predictions and masks are given in [0, 255]; masks are binarized at 128 and
predictions are rescaled to [0, 1] (min-max normalized when not constant).
Fmeasure uses beta^2 = 0.3; its adaptive value thresholds at twice the mean
prediction, and F_avg is the mean of the 256-threshold curve.
"""
import numpy as np

_EPS = np.spacing(1)
_TYPE = np.float64


def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    gt = gt > 128
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


def _get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    return min(2 * matrix.mean(), max_value)


class Fmeasure(object):
    def __init__(self, beta: float = 0.3):
        self.beta = beta
        self.precisions = []
        self.recalls = []
        self.adaptive_fms = []
        self.changeable_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)
        adaptive_fm = self.cal_adaptive_fm(pred=pred, gt=gt)
        self.adaptive_fms.append(adaptive_fm)
        precisions, recalls, changeable_fms = self.cal_pr(pred=pred, gt=gt)
        self.precisions.append(precisions)
        self.recalls.append(recalls)
        self.changeable_fms.append(changeable_fms)

    def cal_adaptive_fm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        binary_prediction = pred >= adaptive_threshold
        area_intersection = binary_prediction[gt].sum()
        if area_intersection == 0:
            adaptive_fm = 0
        else:
            pre = area_intersection / np.count_nonzero(binary_prediction)
            rec = area_intersection / np.count_nonzero(gt)
            adaptive_fm = (1 + self.beta) * pre * rec / (self.beta * pre + rec)
        return adaptive_fm

    def cal_pr(self, pred: np.ndarray, gt: np.ndarray) -> tuple:
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_hist, _ = np.histogram(pred[gt], bins=bins)
        bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0)
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)
        tps = fg_w_thrs
        ps = fg_w_thrs + bg_w_thrs
        ps[ps == 0] = 1
        t = max(np.count_nonzero(gt), 1)
        precisions = tps / ps
        recalls = tps / t
        numerator = (1 + self.beta) * precisions * recalls
        denominator = np.where(numerator == 0, 1, self.beta * precisions + recalls)
        changeable_fms = numerator / denominator
        return precisions, recalls, changeable_fms

    def get_results(self) -> dict:
        adaptive_fm = np.mean(np.array(self.adaptive_fms, _TYPE))
        changeable_fm = np.mean(np.array(self.changeable_fms, dtype=_TYPE), axis=0)
        precision = np.mean(np.array(self.precisions, dtype=_TYPE), axis=0)
        recall = np.mean(np.array(self.recalls, dtype=_TYPE), axis=0)
        return dict(fm=dict(adp=adaptive_fm, curve=changeable_fm), pr=dict(p=precision, r=recall))


class MAE(object):
    def __init__(self):
        self.maes = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)
        self.maes.append(np.mean(np.abs(pred - gt)))

    def get_results(self) -> dict:
        mae = np.mean(np.array(self.maes, _TYPE))
        return dict(mae=mae)


class Smeasure(object):
    def __init__(self, alpha: float = 0.5):
        self.sms = []
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)
        self.sms.append(self.cal_sm(pred, gt))

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        y = np.mean(gt)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            sm = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
            sm = max(0, sm)
        return sm

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        return u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)

    def s_object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x = np.mean(pred[gt == 1])
        sigma_x = np.std(pred[gt == 1], ddof=1)
        return 2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS)

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x, y = self.centroid(gt)
        part_info = self.divide_with_xy(pred, gt, x, y)
        w1, w2, w3, w4 = part_info['weight']
        pred1, pred2, pred3, pred4 = part_info['pred']
        gt1, gt2, gt3, gt4 = part_info['gt']
        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)
        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def centroid(self, matrix: np.ndarray) -> tuple:
        h, w = matrix.shape
        if matrix.sum() == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            area_object = np.sum(matrix)
            row_ids = np.arange(h)
            col_ids = np.arange(w)
            x = np.round(np.sum(np.sum(matrix, axis=0) * col_ids) / area_object)
            y = np.round(np.sum(np.sum(matrix, axis=1) * row_ids) / area_object)
        return int(x) + 1, int(y) + 1

    def divide_with_xy(self, pred: np.ndarray, gt: np.ndarray, x, y) -> dict:
        h, w = gt.shape
        area = h * w
        gt_lt = gt[0:y, 0:x]
        gt_rt = gt[0:y, x:w]
        gt_lb = gt[y:h, 0:x]
        gt_rb = gt[y:h, x:w]
        pred_lt = pred[0:y, 0:x]
        pred_rt = pred[0:y, x:w]
        pred_lb = pred[y:h, 0:x]
        pred_rb = pred[y:h, x:w]
        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = 1 - w1 - w2 - w3
        return dict(gt=(gt_lt, gt_rt, gt_lb, gt_rb), pred=(pred_lt, pred_rt, pred_lb, pred_rb), weight=(w1, w2, w3, w4))

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        h, w = pred.shape
        n = h * w
        x = np.mean(pred)
        y = np.mean(gt)
        sigma_x = np.sum((pred - x) ** 2) / (n - 1)
        sigma_y = np.sum((gt - y) ** 2) / (n - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (n - 1)
        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)
        if alpha != 0:
            score = alpha / (beta + _EPS)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score

    def get_results(self) -> dict:
        sm = np.mean(np.array(self.sms, dtype=_TYPE))
        return dict(sm=sm)


class Emeasure(object):
    def __init__(self):
        self.adaptive_ems = []
        self.changeable_ems = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)
        self.gt_fg_numel = np.count_nonzero(gt)
        self.gt_size = gt.shape[0] * gt.shape[1]
        self.changeable_ems.append(self.cal_changeable_em(pred, gt))
        self.adaptive_ems.append(self.cal_adaptive_em(pred, gt))

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        return self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold)

    def cal_changeable_em(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        return self.cal_em_with_cumsumhistogram(pred, gt)

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        binarized_pred = pred >= threshold
        fg_fg_numel = np.count_nonzero(binarized_pred & gt)
        fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)
        fg_numel = fg_fg_numel + fg_bg_numel
        bg_numel = self.gt_size - fg_numel

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg_numel
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg_numel
        else:
            parts_numel, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel,
                fg_bg_numel=fg_bg_numel,
                pred_fg_numel=fg_numel,
                pred_bg_numel=bg_numel,
            )
            results_parts = []
            for part_numel, combination in zip(parts_numel, combinations):
                align_matrix_value = 2 * (combination[0] * combination[1]) / (
                    combination[0] ** 2 + combination[1] ** 2 + _EPS
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts.append(enhanced_matrix_value * part_numel)
            enhanced_matrix_sum = sum(results_parts)

        return enhanced_matrix_sum / (self.gt_size - 1 + _EPS)

    def cal_em_with_cumsumhistogram(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
        fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
        fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)
        fg_numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
        bg_numel_w_thrs = self.gt_size - fg_numel_w_thrs

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg_numel_w_thrs
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg_numel_w_thrs
        else:
            parts_numel_w_thrs, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel_w_thrs,
                fg_bg_numel=fg_bg_numel_w_thrs,
                pred_fg_numel=fg_numel_w_thrs,
                pred_bg_numel=bg_numel_w_thrs,
            )
            results_parts = np.empty(shape=(4, 256), dtype=np.float64)
            for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):
                align_matrix_value = 2 * (combination[0] * combination[1]) / (
                    combination[0] ** 2 + combination[1] ** 2 + _EPS
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts[i] = enhanced_matrix_value * part_numel
            enhanced_matrix_sum = results_parts.sum(axis=0)

        return enhanced_matrix_sum / (self.gt_size - 1 + _EPS)

    def generate_parts_numel_combinations(self, fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel):
        bg_fg_numel = self.gt_fg_numel - fg_fg_numel
        bg_bg_numel = pred_bg_numel - bg_fg_numel
        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

        mean_pred_value = pred_fg_numel / self.gt_size
        mean_gt_value = self.gt_fg_numel / self.gt_size

        demeaned_pred_fg_value = 1 - mean_pred_value
        demeaned_pred_bg_value = 0 - mean_pred_value
        demeaned_gt_fg_value = 1 - mean_gt_value
        demeaned_gt_bg_value = 0 - mean_gt_value

        combinations = [
            (demeaned_pred_fg_value, demeaned_gt_fg_value),
            (demeaned_pred_fg_value, demeaned_gt_bg_value),
            (demeaned_pred_bg_value, demeaned_gt_fg_value),
            (demeaned_pred_bg_value, demeaned_gt_bg_value),
        ]
        return parts_numel, combinations

    def get_results(self) -> dict:
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=_TYPE))
        changeable_em = np.mean(np.array(self.changeable_ems, dtype=_TYPE), axis=0)
        return dict(em=dict(adp=adaptive_em, curve=changeable_em))


class MCODEvaluator(object):
    """Accumulates S_m, adaptive E, adaptive F, F_avg and MAE over images."""

    def __init__(self):
        self.sm = Smeasure()
        self.em = Emeasure()
        self.fm = Fmeasure()
        self.mae = MAE()

    def step(self, pred255: np.ndarray, gt255: np.ndarray):
        self.fm.step(pred=pred255, gt=gt255)
        self.sm.step(pred=pred255, gt=gt255)
        self.em.step(pred=pred255, gt=gt255)
        self.mae.step(pred=pred255, gt=gt255)

    def results(self) -> dict:
        fm = self.fm.get_results()['fm']
        em = self.em.get_results()['em']
        e_curve = np.asarray(em['curve'], dtype=np.float64)
        f_curve = np.asarray(fm['curve'], dtype=np.float64)
        return {
            'S_m': float(self.sm.get_results()['sm']),
            'E_adp': float(em['adp']),
            'F_adp': float(fm['adp']),
            'F_avg': float(f_curve.mean()),
            'MAE': float(self.mae.get_results()['mae']),
            'E_avg': float(e_curve.mean()),
            'E_max': float(e_curve.max()),
            'F_max': float(f_curve.max()),
        }
