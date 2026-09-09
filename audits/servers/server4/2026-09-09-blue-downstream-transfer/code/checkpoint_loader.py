"""검증된 pretrained W0 모델에 선택 weight만 복사하는 참조 함수. 모델 생성/forward 없음."""
import hashlib
def tensor_sha(t):
    import torch
    x=t.detach().contiguous().cpu()
    h=hashlib.sha256(str((str(x.dtype),list(x.shape))).encode())
    raw=x.view(torch.uint8).numpy().reshape(-1)
    for i in range(0,raw.size,8<<20):h.update(memoryview(raw[i:i+(8<<20)]))
    return h.hexdigest()
def read_selected(path,manifest_entry):
    import torch
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    assert h.hexdigest()==manifest_entry['file']['sha256']
    cp=torch.load(path,map_location='cpu',weights_only=True)
    assert set(cp)=={'weights','cache_c','metadata'}
    assert set(cp['weights'])==set(manifest_entry['weights'])
    for k,t in cp['weights'].items():
        m=manifest_entry['weights'][k]
        assert list(t.shape)==m['shape'] and str(t.dtype)==m['dtype'] and tensor_sha(t)==m['sha256']
    assert tensor_sha(cp['cache_c'])==manifest_entry['method_state']['sha256']
    return cp
def apply_selected_to_verified_w0(model,checkpoint,manifest_entry):
    """호출자: exact base/tokenizer manifest 확인 및 checkpoint마다 W0 복원 필수."""
    import torch
    params=dict(model.named_parameters())
    with torch.no_grad():
        for key,t in checkpoint['weights'].items():
            p=params[key]
            assert p.dtype==t.dtype and p.shape==t.shape
            p.copy_(t.to(p.device))
            assert tensor_sha(p)==manifest_entry['weights'][key]['sha256']
    # cache_c는 AlphaEdit 편집 history. 평가 forward/모델 parameter에 적용하지 않는다.
    # checkpoint metadata의 RNG/context도 downstream prompt/config로 자동 적용하지 않는다.
