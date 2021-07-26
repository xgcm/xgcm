import pytest

from xgcm.grid_ufunc import _parse_grid_ufunc_signature


class TestParseGridUfuncSignature:
    @pytest.mark.parametrize("signature, exp_in_core, exp_out_core, exp_in_ax_pos, exp_out_ax_pos",
                             [("()->()", [()], [()], [()], [()]),
                              ("(X:center)->()", [("X",)], [()], [("center",)], [()]),
                              ("()->(X:left)", [()], [("X",)], [()], [("left",)]),
                              ("(X:center)->(X:left)", [("X",)], [("X",)], [("center",)], [("left",)]),
                              ("(X:left)->(Y:center)", [("X",)], [("Y",)], [("left",)], [("center",)]),
                              ("(X:left)->(Y:center)", [("X",)], [("Y",)], [("left",)], [("center",)]),
                              ("(X:left),(X:right)->(Y:center)", [("X",), ("X",)], [("Y",)], [("left",), ("right",)], [("center",)]),
                              ("(X:center)->(Y:inner),(Y:outer)", [("X",)], [("Y",), ("Y",)], [("center",)], [("inner",), ("outer",)]),
                              ("(X:center,Y:center)->(Z:center)", [("X", "Y")], [("Z",)], [("center", "center")], [("center",)])])
    def test_parse_valid_signatures(self, signature, exp_in_core, exp_out_core, exp_in_ax_pos, exp_out_ax_pos):
        in_core, out_core, in_ax_pos, out_ax_pos = _parse_grid_ufunc_signature(
            signature)
        assert in_core == exp_in_core
        assert out_core == exp_out_core
        assert in_ax_pos == exp_in_ax_pos
        assert out_ax_pos == exp_out_ax_pos

    @pytest.mark.parametrize("signature",
                             ["(x:left)(y:left)->()",
                              "(x:left),(y:left)->",
                              "((x:left))->(x:left)",
                              "((x:left))->(x:left)",
                              "(x:left)->(x:eft),"
                              "(i)->(i)",
                              "(X:centre)->()"])
    def test_invalid_signatures(self, signature):
        with pytest.raises(ValueError):
            _parse_grid_ufunc_signature(signature)
