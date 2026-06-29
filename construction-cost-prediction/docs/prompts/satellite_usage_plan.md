## 実験計画
notebooks/initial_model.ipynbをベースとして、衛星データを追加することにより、精度向上を図ります。
そのために、まずはどのような変数が建設コストconstruction_cost_per_m2_usdに影響するかを洗い出したいです。
衛星データは@docs/prompts/make_intial_model.mdでも説明している通り、Sentinel-2のL2Aを使用しています。
バンド: 「B1」、「B2」、「B3」、「B4」、「B5」、「B6」、「B7」、「B8」、「B8A」、「B9」、「B11」、「B12」を組み合わせてどのような指数を作成すると建設コストにつながる変数を作成できるでしょうか。
例えば、人口の建設物が密集すると建設コストが高くなることを想定して、そのような物体に反応する指数を候補にするといったことが考えられそうです。
ただし、これは指数に限りません。
このような仮説を置いて、衛星データの活用案を複数考えてドキュメントを作成してください。


## 追加ディスカッション
(1)docs/satellite_feature_plan.mdで挙げていただいた指数を計算しましょう。
@data/train_dataset_82ddf14911a54c729380209510ae25ac/train_composite以下のtifについて、指数を計算し、バンドを追加してほしいです。
その際、descriptionsに追加した順番でバンド名の要素を追加してほしいです。
バンドを追加した画像は、@data/train_dataset_82ddf14911a54c729380209510ae25ac/train_composite-add-featuresに保存してください。
その実装は、notebooks/add-bands.ipynbに作成します。

(2)先ほど、(1)途中でno left on deviceになりましたので、処理を停止し、データを削除しました。
実装内容を以下に変更します。
@data/train_dataset_82ddf14911a54c729380209510ae25ac/train_composite以下のsentinel_2_*.tifについて、docs/satellite_feature_plan.mdで挙げていただいた指数を計算してほしいです。
バンドを計算したあと、画像は保存せず、対応するマスターテーブル(data/train_dataset_82ddf14911a54c729380209510ae25ac/train_tabular.csv)に特徴量カラムとして追加してほしいです。
sentinel_2_*.tifから計算される指数については、マスターテーブルのsentinel2_tiff_file_nameに対応するファイル名が記載されています。
さらに、VIIRSの画像についても特徴量を追加します。これは、data/train_dataset_82ddf14911a54c729380209510ae25ac/train_composite以下にviirs_*.tifとして格納されています。
あらたな指数にはしません。マスターテーブルのviirs_tiff_file_nameに対応するファイル名が記載されています。

これらのバンド情報について、ラスター画像を以下の値で集約します。
パーセンタイル0,10,20,30,40,50,60,70,80,90,100、平均、標準偏差

特徴量は以下のカラム名で作成してほしいです。
{band_name}_{集約方法}

たとえば、"B1_pct0"、"B1_pct10"、"B1_mean"

その実装は、notebooks/create-add-features.ipynbに作成します。



