"""Gradient clipping with explicit distinction between NaN/Inf and norm overflow."""
import math
import torch


class GradientFailure(FloatingPointError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


@torch.no_grad()
def gradient_report(named_parameters):
    rows, norms = [], []
    bad = norm_overflow = False
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite = torch.isfinite(gradient)
        nonfinite = int((~finite).sum())
        norm = None if nonfinite else float(gradient.double().norm())
        if norm is not None and not math.isfinite(norm):
            norm_overflow = True
            norm = None
        finite_values = gradient[finite]
        maximum = float(finite_values.abs().max()) if finite_values.numel() else None
        rows.append(dict(name=name, shape=list(gradient.shape), nonfinite=nonfinite,
                         norm_fp64=norm, max_finite_abs=maximum))
        bad |= bool(nonfinite)
        if norm is not None:
            norms.append(norm)
    total = math.hypot(*norms) if not bad and not norm_overflow else None
    return dict(all_gradients_finite=not bad, total_norm_fp64=total,
                nonfinite_parameters=[r['name'] for r in rows if r['nonfinite']],
                parameters=rows)


@torch.no_grad()
def clip_gradients(named_parameters, max_norm):
    """Return (unclipped norm, recovered FP32 norm overflow).

    Never suppress true non-finite gradients. Use FP64 only on the exceptional
    finite-gradient overflow path; normal batches retain PyTorch's fast path.
    """
    if max_norm <= 0 or math.isnan(max_norm):
        raise ValueError('Gradient cap must be positive')
    named_parameters = list(named_parameters)
    parameters = [p for _, p in named_parameters if p.grad is not None]
    try:
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
        return float(norm), False
    except RuntimeError as exc:
        if 'non-finite' not in str(exc):
            raise
        report = gradient_report(named_parameters)
        if not report['all_gradients_finite']:
            report['classification'] = 'nonfinite_gradient_entries'
            raise GradientFailure('NaN/Inf gradient entries: '+', '.join(report['nonfinite_parameters'][:5]), report) from exc
        norm = report['total_norm_fp64']
        report['classification'] = 'finite_gradients_fp32_norm_overflow'
        if not math.isfinite(max_norm) or norm is None or not math.isfinite(norm):
            raise GradientFailure('Gradient norm overflow; safe recovery requires a finite clipping cap', report) from exc
        coefficient = min(max_norm/(norm+1e-6), 1.)
        for parameter in parameters:
            # Keep the coefficient multiplication in FP64 to avoid FP32
            # underflow when normalizing exceptionally large finite gradients.
            parameter.grad.copy_((parameter.grad.double()*coefficient).to(parameter.grad.dtype))
        return norm, True
