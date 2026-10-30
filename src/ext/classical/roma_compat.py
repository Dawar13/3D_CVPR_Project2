"""romatch API shim for the Barath repo.

The repo's AC extractor (rs_ac_extractor.extract_affine_correspondences_roma) expects the *unbatched* RoMa output
of the original Parskatt/RoMa package: warp (H, W', 4), certainty (H, W'). PyPI romatch 0.1.x only supports
batched inference and returns warp (1, H, W', 4), certainty (1, H, W'), which makes the extractor raise IndexError
(every pair -> 0 correspondences). This shim drops the batch dimension so the repo code runs unchanged; nothing else
in RoMa changes. (romatch 0.1.2 roma_outdoor also needs the fused-local-corr package, installed separately.)
"""
import romatch


class _Unbatched:
    def __init__(self, model):
        self._m = model

    def match(self, *a, **k):
        warp, cert = self._m.match(*a, **k)
        assert warp.shape[0] == 1
        return warp[0], cert[0]

    def __getattr__(self, name):
        return getattr(self._m, name)


def roma_outdoor_unbatched(*a, **k):
    return _Unbatched(_orig(*a, **k))


_orig = romatch.roma_outdoor


def patch():
    romatch.roma_outdoor = roma_outdoor_unbatched
