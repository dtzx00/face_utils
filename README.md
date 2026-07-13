# face_utils

A small Python toolkit for facial-feature extraction, image preprocessing, and the statistical analyses used in our facial-image research.

The package wraps common steps in a facial-image analysis pipeline: pulling landmarks and attributes from the [Face++](https://www.faceplusplus.com/) API, aligning and normalizing face crops, and running the classification, matching, and AUC-comparison routines used in the accompanying papers.

## Installation

```bash
pip install -e .
```

Requires Python 3.6+. Dependencies are installed automatically (`requests`, `numpy`, `scipy`, `pandas`, `opencv-python-headless`, `scikit-learn`, `matplotlib`, `statsmodels`, `python-dotenv`).

Face++ calls require API credentials. Put them in a local `.env` file (never commit it):

```
FPP_KEY=your_key
FPP_SECRET=your_secret
```

## Modules

### `face_utils.utils` — image + API pipeline
- **Face++ extraction:** `get_faceplusplus_outputs`, `Get_FacePlusPlus_Outputs`, `Get_Flattened_Dict`, `get_rectangle`, `load_api_keys`
- **Preprocessing:** `get_rotated_image` (eye-based alignment), `to_grayscale`, `Get_Euclidean_Distance`
- **Modeling / matching:** `Classify_LR`, `logistic_regression`, `match_df`, `match_age`, `match_attributes`, `get_dist_angle`
- **Data handling:** `load_data`, `clean_attributes`, `split_attributes`
- **Augmentation / models:** `augment_img`, `load_custom_vgg`, `load_custom_lr`

### `face_utils.stats` — statistics helpers
- **Formatting:** `round_str`, `convert_ci_str`, `stars`, `pval`
- **Effect sizes / CIs:** `cohen_effect_size`, `get_confidence_interval_data`, `odds_prob`
- **DeLong AUC comparison:** `delong_roc_test`, `delong_roc_variance`, `fastDeLong`, `calc_pvalue`, `get_confidence_interval`

## Quick start

```python
from face_utils import utils, stats

# extract Face++ landmarks + attributes for an image
key, secret = utils.load_api_keys('.env')
score, landmarks, attributes, rectangle, status = utils.get_faceplusplus_outputs(img, (key, secret))

# align and normalize a detected face
aligned = utils.get_rotated_image(img, left_eye, right_eye)

# compare two ROC curves (DeLong test)
p_value = stats.delong_roc_test(ground_truth, predictions_model_a, predictions_model_b)
```

## Selected bibliography

This toolkit supports research on facial-image analysis. Related work:

- Gheorghiu, A. I., Callan, M. J., & Skylark, W. J. (2017). Facial appearance affects science communication. *Proceedings of the National Academy of Sciences, 114*(23), 5970–5975. https://doi.org/10.1073/pnas.1620542114
- Wang, D., Nair, K., Kouchaki, M., Zajac, E. J., & Zhao, Y. (2019). A case of evolutionary mismatch? Why facial width-to-height ratio may not predict behavioral tendencies. *Psychological Science, 30*(7), 1074–1081. https://doi.org/10.1177/0956797619849928
- Kosinski, M. (2017). Facial width-to-height ratio does not predict self-reported behavioral tendencies. *Psychological Science, 28*(11), 1675–1682. https://doi.org/10.1177/0956797617716929
- Wang, Y., & Kosinski, M. (2018). Deep neural networks are more accurate than humans at detecting sexual orientation from facial images. *Journal of Personality and Social Psychology, 114*(2), 246–257. https://doi.org/10.1037/pspa0000098
- Wang, D. (2022). Presentation in self-posted facial images can expose sexual orientation: Implications for research and privacy. *Journal of Personality and Social Psychology, 122*(5), 806–824. https://doi.org/10.1037/pspa0000294

## License

BSD 2-Clause.
