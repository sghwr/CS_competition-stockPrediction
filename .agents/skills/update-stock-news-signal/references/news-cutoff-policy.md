# News Cutoff Policy

## Evidence hierarchy

Prefer sources in this order:

1. Exchange filings, listed-company announcements, regulator releases, and official statistics.
2. Original government or industry-body publications.
3. Reputable financial news carrying a visible publication timestamp and attributable facts.
4. Secondary analysis only when its publication time and underlying evidence are both traceable.

Do not use social posts, unattributed aggregations, search snippets, generated summaries, or pages with no verifiable publication date as hard-rule evidence.

## Search process

1. Resolve the cutoff in Asia/Shanghai time before searching.
2. Search broad market state first, then industry catalysts, then named companies and sector leaders.
3. Record every material query, including queries that return no accepted source.
4. Open the source page. Do not treat the search-result date or snippet as verification.
5. Record title, canonical URL, publisher, publication time, retrieval time, query, affected entities, and the exact claim used.
6. Cross-check market-moving claims against a primary source when available.
7. Reject any item published after the cutoff, even if it describes an earlier event.

Useful query families include:

```text
<cutoff month> 中国 宏观 政策 官方
<industry> 政策 公告 截止 <cutoff date>
<stock code or company> 公告 <cutoff date>
site:sse.com.cn <stock code> 公告
site:szse.cn <stock code> 公告
```

Adapt terms to the requested date. Never add a later date to obtain retrospective explanations.

## Timestamp rules

- Accept an exact timestamp only when the page exposes it; retain its timezone.
- Accept a date-only publication only when that date is on or before the cutoff date. Do not invent a time.
- If a page shows both publication and update times, use the original version only when its relevant content can be shown to have existed before the cutoff. Otherwise reject it.
- When a page is mutable, prefer an archived or versioned source.
- A source retrieved after the cutoff can still be admissible if its pre-cutoff publication is verifiable and no post-cutoff edit is being relied on.

## Rule construction

Separate three layers:

- Market regime: bull, bear, or sideways evidence and its expected effect on exposure.
- Industry linkage: sector-level catalyst or risk with a bounded prior adjustment.
- Leader effect: named leader evidence and a bounded effect on related stocks.

Keep adjustments small relative to model score dispersion. Use blocking only for explicit, severe risks. Preserve the base portfolio. If evidence is weak or conflicting, disable the manual layer rather than forcing a directional view.

Write the natural-language thesis before editing policy JSON. Every nonzero industry or stock prior must map to at least one accepted manifest item or an explicitly declared date-independent prior.
