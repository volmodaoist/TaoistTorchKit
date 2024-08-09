'''
TODO # 封装 cleverhans/foolbox

 1. 对抗攻击领域必备网站，列举了当前各种模型性能的排行榜/文章:
    https://robustbench.github.io/#leaderboard
 
 2. 对抗攻击当前通常使用 AutoAttack 作为 benchmark:
    https://github.com/fra31/auto-attack

'''

from .Attacker import ClassifierAttackerFb