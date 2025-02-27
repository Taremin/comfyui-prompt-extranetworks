# ComfyUI Prompt ExtraNetworks

これは [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 用のカスタムノードです。
`LoraLoader` や `HypernetworkLoader` の代わりにプロンプトを受け取って、プロンプト内の指定によって LoRA や HN を読み込み適用します。
このカスタムノードの主目的は、プロンプトがランダムに変更されるときなどに `LoraLoader` ノードを繋ぎ直さずに変更できるようにすることです。

## 機能

### LoRA

以下の文法をプロンプトに含めることでLoRAを読み込みます。

```
<lora:lora-filename[:model_strength[:clip_strength]][:cache={always|once|none}]>
```

- lora-filename: LoRAのファイル名。拡張子も含めて指定してください。(LoraLoaderで指定するファイル名と同じ)
- model_strength: モデルへの適用率。省略した場合は `1.0` になります。
- clip_strength: CLIPへの適用率。省略した場合は `model_strength` と同じになります。
- cache: LoRAのロードをキャッシュします。デフォルトは `none` です。
  - always: 常にキャッシュを行います。
  - once: 一度だけキャッシュを行います。
  - none: キャッシュを行いません。

#### LoRA Block Weight

[ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) をインストールしている場合、以下のように書くことで LoRA Block Weight を使用することが出来ます。

```
<lora:lora-filename[:model_strength[:clip_strength]][:lbw=lbw_preset][:cache={always|once|none}]>
```

- lbw_preset: `SDXL-ALL` などのプリセット名か `1,0,0,0,0,0,1,1,1,1,1,1` など直接ウェイト表記
  - ComfyUI-Inspire-Pack で使用できるプリセット名が使用できます

#### キャッシュについて

十分速いストレージの場合は劇的な効果は期待できません。
また、OSのファイルシステムキャッシュが効いている場合などほとんど効果が見られない場合もあります。
キャッシュを使用するにしても `always` は大体の場合不要で、`once` で十分なことが多いはずです。
(`once` を指定しても毎回読み込まれる LoRA は毎回更新されてキャッシュされ続けることになるため)

### HyperNetwork

以下の文法をプロンプトに含めることでHyperNetworkを読み込みます。

```
<hypernet:hn-filename[:strength]>
```

- hn-filename: HyperNetworkのファイル名。拡張子も含めて指定してください。（HypernetworkLoaderで指定するファイル名と同じ）
- strength: モデルへの適用率。

**この機能はテストされていません。**

## その他

[WebUI Monaco Prompt](https://github.com/Taremin/webui-monaco-prompt) のスニペットに対応しています。

## ライセンス

[MIT](./LICENSE)
