# MUFG2023  5th place solution 🥇

[link](https://signate.jp/competitions/1088)

## 課題
クレジットカードの顧客登録情報や決済手段・利用場所といった定量及び定性データを元に分析モデルを構築して、カード不正利用の検知すること。 </br>

### 評価関数
F1Score
</br>

</br>

## Work

![全体像](/png/MUFG2023SubImage.png)
</br>
</br>
</br>

### モデル
- (deberta-v3-base) </br>  ↓  </br>
- catboost

</br>

### 前処理
- state,cityの出現率の低いものに対してRareエンコーディング
- カテゴリ変数をラベルエンコーディング
- WoEエンコーディング  (WoE = log( 陽性例の割合 / 陰性例の割合 ))
   1. 目的変数を反転(1を0、0を1) </br> > neg_y = pd.Series( np.where( y == 1, 0, 1))
    1. 目的変数が0と1の総数を計算  </br> > total_pos = y.sum() </br> >  total_neg = neg_y.sum()
    1. WoEの計算式の分母、分子を計算  </br> > pos = train.groupby("カテゴリ変数")[target].sum() / total_pos </br> >  neg = train.groupby("カテゴリ変数")["target_neg"].sum() / total_neg
    1. WoEを計算  </br> > woe = np.log( pos / neg ) 
    1. カテゴリ変数をWoEに置き換える
    </br>

    
### deberta-v3-base
exp021 ( cv : 0.5357 )
- input text : "merchant" + "[SEP]" + "train(test)ファイルの変数名" + ... + "card" + "[SEP]" + "cardファイルの変数名" + ... + "user" + "[SEP]" + "userファイルの変数名" + ...
- Last Hidden State Output : Mean Pooling
- epoch : 4


</br>
</br>
</br>


## Final Submission
exp024 (catboost)

CV : 0.665919

Public LB : 0.6769145

Private LB : 0.6787937


</br>
</br>
</br>

### うまくいかなかったこと、やらなかったこと
- 分均衡データの扱い
   - FocalLoss、クラスラベルの重みの設定を行ったが精度の向上はなかった
- アンサンブル
   - blending、stackingを行ったがlocal cvでオーバーフィットしているようだった
   - 複数モデルの予測値の集約変数を作ってモデルに組み込んだが精度が落ちた
- ハイパラメータのチューニング 
