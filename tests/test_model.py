import pytest

torch = pytest.importorskip("torch")

from ed_dscr.models.ed_dscr import EDDSCR
from ed_dscr.models.losses import total_loss


def test_training_and_rgb_only_inference_contract():
    model = EDDSCR(6, backbone="tiny", channels=32, pretrained=False, ddg_base_channels=8)
    image = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 6, (2, 64, 64))
    normal = torch.nn.functional.normalize(torch.randn(2, 3, 64, 64), dim=1)
    model.train(); output = model(image, target, normal)
    losses = total_loss(output, target, normal)
    assert output["logits"].shape == (2, 6, 64, 64)
    assert "edar_loss" in output and torch.isfinite(losses["loss"])
    model.eval()
    with torch.no_grad(): inference = model(image)
    assert "edar_loss" not in inference
    assert inference["normal"].shape == (2, 3, 64, 64)

