# 'SynthTIGER'
## Configuration
- Template
    - 템플릿 클래스의 `init_save()`: 생성된 데이터를 저장하기 위한 환경 설정
- Text Style (`style`)
    - `prob`: text style 적용 확률
    - `class Switch`:
        - `state`: text style의 존재 여부
```yaml
coord_output: true
mask_output: true
glyph_coord_output: true
glyph_mask_output: true

vertical: false
quality: [50, 95]
visibility_check: true

midground: 0.25 # midground를 적용할 확률
midground_offset:
  percents: [[-0.5, 0.5], [-0.5, 0.5]] # midground 확대 / 축소 비율
foreground_mask_pad: 5 # midground 생성 시 대충 foreground가 차지하는 영역에는 텍스트가 렌더링되지 않도록 비워두는데, 얼마 만큼의 여유치를 두고 비워둘지는 정합니다.

corpus:
    # `weights`: 예를 들어 `[1, 3, 2]`라고 하면 각 확률은 [`1 / 6, 3 / 6, 2 / 6]`이 됩니다. 이 확률 분포를 가지고 각 확률에 해당하는 요소를 샘플링합니다.
  weights: [1, 0]
  args:
    ### Length augmentable corpus
    - paths: [/Users/jongbeomkim/Desktop/workspace/scene_text_image_generator/synthtiger/resources/corpus/ko_corpus.txt]
      weights: [1]
      min_length: 1
      max_length: 50
      textcase: [lower, upper, capitalize]
      augmentation: 0.1
      augmentation_length: [1, 25]
    ### Character augmentable corpus
    - paths: []
      weights: []
      min_length: 1
      max_length: 25
      textcase: [lower, upper, capitalize]
      augmentation: 0
      augmentation_charset: resources/charset/alphanum_special.txt

font:
  paths: [resources/font]
  weights: [1]
  size: [30, 80] # Font size range
  bold: 0.5 # 볼드 적용 확률

texture:
  prob: 0.5 # 배경 이미지의 texture 적용 확률. 배경 이미지의 texture가 적용되지 않으면 배경은 단색입니다.
  args:
    paths: [resources/image]
    weights: [1] # `paths`에서 지정된 각 디렉토리를 선택할 확률
    alpha: [0, 1] # 배경 이미지의 texture의 RGBA color model의 alpha 값
    grayscale: 0 # Grayscale로 변환할 확률
    crop: 1

# "/resources/colormap/iiit5k_gray.txt" 파일의 각 행은 4개 또는 6개의 숫자로 이루어져 있으며 이는 (평균, 표쥰편차)가 2개 또는 3개 연속한 것입니다. ("It usually consists of 2, or 3 clusters with the mean gray-scale colors and their standard deviation (s.t.d).")
colormap2:
  paths: [/Users/jongbeomkim/Desktop/workspace/scene_text_image_generator/synthtiger/resources/colormap/iiit5k_gray.txt]
  weights: [1]
    # 2개의 clusters를 가진 컬러를 사용할 지 아니면 3개의 clusters를 가진 컬러를 사용할 지를 의미합니다.
    # 2개의 clusters를 사용한다면 foreground color and background color의 2가지 컬러를 샘플링하는 것이고, 3개의 clusters를 사용한다면 foreground color, background color and style color의 3가지 컬러를 샘플링하는 것입니다.
  k: 2
    # RGBA color model의 alpha 값을 샘플링하는 데 사용되며, `alpha[0]` 이상 `alpha[1]` 미만의 실수 중 하나를 uniform distribution에서 샘플링합니다. ("synthtiger/components/color/gray_map.py": `alpha = np.random.uniform(self.alpha[0], self.alpha[1])`)
    # alpha 값이 0일 경우 text border를 제외한 배경이 그대로 비춰 보입니다.
  alpha: [1, 1]
    # "synthtiger/utils/image_util.py": `to_rgb()`를 적용할 확률
    # `to_rgb()`를 적용하지 않으면 `rgb = (gry, gray, gray`)를 사용하고, 적용하면 랜덤 샘플링한 RGB 값을 사용합니다.
  colorize: 1 # grayscale image를 사용하지 않고 colored image를 사용할 확률

colormap3:
  paths: [/Users/jongbeomkim/Desktop/workspace/scene_text_image_generator/synthtiger/resources/colormap/iiit5k_gray.txt]
  weights: [1]
  k: 3
  alpha: [1, 1]
  colorize: 1

color:
  gray: [0, 255]
  alpha: [1, 1]
  colorize: 1

shape:
  prob: 1
  args:
    weights: [1, 1]
    args:
      # elastic distortion
      - alpha: [15, 30]
        sigma: [4, 12]
      # elastic distortion
      - alpha: [0, 2]
        sigma: [0, 0.6]

layout:
  weights: [4, 1]
  args:
    # flow layout
    - space: [-2, 5]
      line_align: [middle]
    # curve layout
    - curve: [20, 40]
      space: [-2, 5]
      convex: 0.5
      upward: 0.5

style:
  prob: 1
  args:
    # weights: [2, 1, 1]
    weights: [0, 1, 0]
    args:
      # Text border
      - size: [1, 6]
        alpha: [1, 1]
        grayscale: 0
      # Text shadow
      - distance: [4, 8]
        angle: [0, 360]
        alpha: [0.3, 0.7]
        grayscale: 0
      # Text extrusion
      - length: [1, 12]
        angle: [0, 360]
        alpha: [1, 1]
        grayscale: 0

transform:
  prob: 1
  args:
    weights: [1, 1, 1, 1, 1, 1, 2]
    args:
      # perspective x
      - percents: [[0.5, 1], [1, 1]]
        aligns: [[0, 0], [0, 0]]
      # perspective y
      - percents: [[1, 1], [0.5, 1]]
        aligns: [[0, 0], [0, 0]]
      # trapezoidate x
      - weights: [1, 0, 1, 0]
        percent: [0.75, 1]
        align: [-1, 1]
      # trapezoidate y
      - weights: [0, 1, 0, 1]
        percent: [0.5, 1]
        align: [-1, 1]
      # skew x
      - weights: [1, 0]
        angle: [0, 30]
        ccw: 0.5
      # skew y
      - weights: [0, 1]
        angle: [0, 10]
        ccw: 0.5
      # rotate
      - angle: [0, 10]
        ccw: 0.5

pad:
  prob: 1
  args:
    pxs: [[0, 10], [0, 10], [0, 10], [0, 10]]

### (e)
postprocess:
  args:
    # Gaussian noise
    - prob: 1
      args:
        scale: [4, 8]
        per_channel: 0
    # Gaussian blur
    - prob: 0.9
      args:
        sigma: [0, 2]
    # Resize
    - prob: 0.1
      args:
        size: [0.4, 0.4]
    # Median blur
    - prob: 1
      args:
        k: [1, 1]
```
## References
- [What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis](https://arxiv.org/pdf/1904.01906.pdf)
- [Synthetic Data for Text Localisation in Natural Images](https://arxiv.org/pdf/1604.06646.pdf)
- [Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition](https://arxiv.org/pdf/1406.2227.pdf)
- [ASTER: An Attentional Scene Text Recognizer with Flexible Rectification]
- [Best practices for convolutional neural networks applied to visual document analysis]
- https://github.com/ankush-me/SynthText
- https://github.com/TianzhongSong/awesome-SynthText
- https://github.com/JarveeLee/SynthText_Chinese_version
- https://github.com/gungui98/SynthText
