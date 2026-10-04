# Data notice for generated packs

Ship this notice with any pack, export or clip made with these tools, and put its two lines in the
pack's `meta` (`source`, `license`). Fill in the brackets.

> **AI-generated animation.** The clips in this pack were generated with NVIDIA Kimodo
> (`Kimodo-SOMA-RP-v1.1`, Hugging Face revision [revision], Kimodo [commit]) from text prompts and
> ground paths, on [date], and converted with Kansei's `kansei-anim-bake/genanim`. The SOMA body and
> skeleton come from NVIDIA's Kimodo repository (Apache-2.0).
>
> The model is licensed under the NVIDIA Open Model License; NVIDIA claims no ownership rights in
> its outputs. The model's text encoder is built on Meta Llama 3 (Llama 3 Community License).
> Copyright in purely AI-generated material may not exist: this pack is offered as is, without a
> claim that the MIT licence or any other licence covers the clips themselves.
>
> [For a pack that also holds third-party clips, e.g. route B on the GASP mannequin: those clips
> keep their own licence and the pack follows the stricter one. A GASP-based pack is private.]

`meta` example:

```json
"meta": {
  "source": "AI-generated with NVIDIA Kimodo-SOMA-RP-v1.1 (rev [revision], Kimodo [commit]) on [date]: prompts and paths in kansei-anim-bake/genanim/prompts; SOMA body (Apache-2.0)",
  "license": "Model: NVIDIA Open Model License; NVIDIA claims no ownership of outputs. AI-generated: no copyright claimed over the clips."
}
```
