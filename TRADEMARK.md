# Trademark and naming policy

The code in this repository is MIT-licensed and the released weights are
CC BY-SA 4.0. **Neither licence grants any rights in the names.** This file
says what the names are and how they may be used, because a copyright licence
is not the instrument that governs them.

## The marks

- **userDevice** — the maker. The mark is not scoped to this repository: it
  is used for userdevice.net and for other software and physical products
  released under that name, and this policy does not narrow it to the model.
- **RTUS** / **userdevice-rtus** — this model family, and the artifact stem
  `userdevice-rtus-<version>-<shape>`.

These are unregistered marks, used in commerce, and asserted as such.

Not claimed: **`rtmosr_ea_film`** and **`rtmosr_ea_film_sd`**, the
architecture class names. They are built on the RTMoSR name, which belongs to
its authors (https://github.com/rewaifu/RTMoSR), and no rights in it are
claimed here.

## What you may do without asking

- Use, modify and redistribute the code and the weights under their
  respective licences. That is the point of publishing them.
- **State factually** that your work uses, is built on, is fine-tuned from,
  or is compared against userDevice RTUS. Accurate reference to a thing by
  its name is not an infringement and is not restricted here — it is
  encouraged, and CC BY-SA 4.0 requires the attribution anyway.
- Reproduce the name in documentation, papers, benchmarks, model cards and
  release notes.

## What needs permission

- **Naming your derivative work `userdevice-*` or `RTUS-*`**, or any name
  close enough that a user could take it for an official release. Fine-tune
  the weights all you like — give the result your own name, and say what it
  was derived from.
- Using **userDevice** or **RTUS** as, or within, your product name, service
  name, company name, or domain.
- Using the marks in a way that **implies endorsement, affiliation, or that
  your work is an official release**.
- Using the marks on a modified model **presented as the original**. This is
  the case the policy exists for: a degraded or retrained model published
  under this name misrepresents measurements that were made on specific
  weights, on named hardware, with a published method.

## Why this matters more than usual for a model

Every quality and latency figure published for RTUS is tied to an exact
checkpoint, an exact degradation recipe, and an exact device. The name is
how someone finds those numbers. If the name travels onto weights the
numbers were not measured on, the numbers become false without anyone
having lied — which is the specific harm this policy prevents.

## Asking

Open an issue on https://github.com/ltomes/userdevice-rtus. Reasonable
requests are expected to be granted; the point is to know, not to refuse.
