# OCS — OneClickSpine

[English](README.md) · **한국어**

캐릭터 일러스트 한 장을 리깅된 **Spine 4.2** 스켈레톤으로 만듭니다.

PNG를 넣으면 레이어로 분리하고, 더미 파츠를 걸러낸 뒤 본을 배치합니다. 그다음 가중 메시 리그를 만들고 아틀라스를 패킹한 뒤, 자체 완결 HTML 미리보기를 돌려줍니다.

```
이미지 → 레이어 분리 → 레이어 정리 → 본 배치 → 팔다리 분할 → 리깅 → Spine + 미리보기
```

| 1. 업로드 | 2. 레이어 정리 |
|:---:|:---:|
| ![업로드](docs/screenshots/01-upload.png) | ![레이어 정리](docs/screenshots/02-layers.png) |
| **3. 본 배치** | **4. 리깅 · 미리보기** |
| ![본 배치](docs/screenshots/03-bones.png) | ![리깅 미리보기](docs/screenshots/04-preview.png) |

<p align="center"><sub>스크린샷은 로컬 테스트 프로젝트입니다.</sub></p>

레이어 분리는 [see-through](https://github.com/shitagaki-lab/see-through)를 git 서브모듈로 사용합니다 (Apache-2.0). [spine-animation-ai](https://github.com/GenielabsOpenSource/spine-animation-ai)는 선행 연구로만 참고했고 **코드는 재사용하지 않습니다**. 자세한 내용은 [NOTICE.md](NOTICE.md)를 보세요.

---

## 요구 사항

- [uv](https://github.com/astral-sh/uv), Python 3.12, git
- 레이어 분리: NVIDIA GPU **또는** Apple Silicon, 모델 가중치용 여유 공간 약 20 GB
- 분리 이후 단계(정리, 본, 리깅, 내보내기, 미리보기)는 CPU에서도 동작합니다

| | |
|---|---|
| **NVIDIA** | 16 GB면 여유롭고, 8 GB는 group offload로 가능합니다. Blackwell은 CUDA 12.8 / torch `cu128`이 필요합니다. |
| **Apple Silicon** | `./scripts/setup_env.sh --with-gpu`. MPS에서는 해상도가 **768**로 제한됩니다. 외장 GPU보다 느립니다. |
| **GPU 없음** | 에디터는 그대로 쓸 수 있습니다. PSD를 가져오거나 [데모 프로젝트](#gpu-없이)를 만드세요. |

헬퍼 스크립트: Windows는 `.ps1`, macOS / Linux는 `.sh`.

## 설치

```bash
git clone --recurse-submodules https://github.com/sleeeppy/OCS.git
cd OCS
```

Windows:

```powershell
./scripts/setup_env.ps1
```

macOS / Linux:

```bash
./scripts/setup_env.sh              # 에디터만, GPU 없음 — 1분 이내
./scripts/setup_env.sh --with-gpu   # torch + see-through 포함
```

`setup_env.ps1`은 CUDA가 없으면 실패합니다 (Windows에서는 보통 설치 문제입니다). `setup_env.sh`는 실패하지 않습니다 — GPU는 레이어 분리에만 필요합니다.

오프라인 미리보기를 쓰려면 아래를 실행하세요. Spine Web Player는 **커밋하지 않습니다** (재배포가 제한되며, 본인 Spine 라이선스가 필요합니다). 이 단계를 건너뛰면 미리보기는 unpkg에서 런타임을 불러옵니다:

```bash
./scripts/fetch_spine_player.sh     # macOS / Linux
./scripts/fetch_spine_player.ps1    # Windows
```

## 실행

```bash
./scripts/run_ocs.sh      # macOS / Linux
./scripts/run_ocs.ps1     # Windows
```

그다음 <http://127.0.0.1:8765/> 를 엽니다.

**투명 배경 PNG**를 쓰세요. OCS는 단색 배경에서도 실루엣을 추정하지만, 실제 컷아웃이 품질에 가장 큰 영향을 줍니다.

GPU 첫 실행은 모델 가중치 약 12 GB를 받습니다. 이후에는 이미지당 몇 분 정도입니다.

### 네 단계

1. **업로드** — 해상도, 스텝, 시드는 *고급 설정*에 있습니다. group offload는 기본으로 켜져 있습니다 (VRAM 약 10 GB). MPS에서는 켜 두세요.
2. **레이어 정리** — 확실한 더미는 자동 제외하고, 애매한 것만 표시합니다. 체크를 풀면 제외되고, 자동 제외된 레이어도 되돌릴 수 있습니다.
3. **본 배치** — 조인트를 드래그하세요. `Shift`+드래그는 하위까지, `X`는 좌우 미러, `S`는 대칭 스냅, `0`은 화면 맞춤입니다.
4. **리깅 · 미리보기** — 가중 메시, 아틀라스, `skeleton.json`, 다운로드 가능한 HTML 미리보기.

`right`는 **캐릭터의** 오른쪽 (보는 사람 기준 왼쪽)입니다. 반대로 잡으면 애니메이션이 전부 뒤집힙니다.

## 결과물

```
workspace/projects/<id>/export/
  skeleton.json     Spine 4.2 — 가중 메시 + 프리셋 애니메이션
  skeleton.atlas    libgdx 아틀라스
  skeleton.png      패킹된 텍스처
  preview.html      단독 실행 (에셋을 data URI로 인라인)
```

프리셋: `idle` (및 변형), `walk`, `wave`, `jump`, `turn_head`.

## GPU 없이

GPU가 필요한 단계는 레이어 분리뿐입니다. see-through 결과가 이미 있거나, 이후 단계만 반복하고 싶다면:

```bash
.venv/bin/python scripts/import_psd.py path/to/foo.psd
```

PSD도 없다면 합성 프로젝트를 만들고 에디터를 여세요:

```bash
.venv/bin/python scripts/make_demo_project.py --all
./scripts/run_ocs.sh
```

## 테스트

```bash
.venv/bin/python -m pytest tests -q          # macOS / Linux
.venv/Scripts/python.exe -m pytest tests -q  # Windows
```

## 라이선스

Apache-2.0. [LICENSE](LICENSE)와 [NOTICE.md](NOTICE.md)를 보세요. Spine Web Player를 쓰려면 Esoteric Software의 라이선스가 필요합니다.
