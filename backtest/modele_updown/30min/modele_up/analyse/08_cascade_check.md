# Test empirique : asymmetrie REELLE des baisses vs hausses BTC

## Setup
- Test set : 34,830 bougies (TEST_DAYS=365)
- Memes 19 features pour les 2 modeles, memes hyperparams

## Symetrie des bougies extremes

| metric | red | green |
|---|---|---|
| |size| max | 3.997% | 3.502% |
| std (only own sign) | 0.1763% | 0.1723% |
| count > 1% | 108 | 82 |
| count > 2% | 4 | 9 |

## Test : le modele voit-il les vraies grosses bougies ?

### TOP 100 BIG RED candles (= les chutes les plus fortes)

- taille moyenne : -1.383%, max chute : -3.997%
- **p_down moyen attribue par le modele DOWN : 48.65%**
- p_down median : 48.26%, max : 56.61%
- p_up moyen (modele UP, miroir) : 51.36%

### TOP 100 BIG GREEN candles (= les rallies les plus forts)

- taille moyenne : 1.341%, max rally : 3.502%
- **p_up moyen attribue par le modele UP : 49.64%**
- p_up median : 49.06%, max : 59.74%
- p_down moyen : 50.36%

## Test : ce que le modele predit comme tres haut, est-ce reellement grand ?

### TOP 100 highest p_up
- p_up moyen : 58.12%
- taille de la bougie suivante : mean=0.0035%, median=0.0388%
- % qui sont effectivement vertes : 54.00%
- bougies vraiment grosses (>0.5%) : 11/100

### TOP 100 highest p_down
- p_down moyen : 56.52%
- taille de la bougie suivante : mean=-0.0498%, median=-0.0552%
- % qui sont effectivement rouges : 54.00%
- bougies vraiment grosses (<-0.5%) : 5/100

