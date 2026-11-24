"""Answer questions about the video itself, which its captions rarely mention."""

import re

from .models import VideoInfo
from .timestamps import format_time

_VIDEO = r"(?:(?:this|the) )?(?:youtube )?(?:video|talk|lecture|podcast|episode|clip)"
_WHAT = r"(?:(?:what|which)(?:s| is| was)? )?(?:the )?"
_LEAD = re.compile(
    r"^(?:(?:hi|hey|hello|ok|okay|so|and|please)\s+)*"
    r"(?:(?:can|could|would) you )?(?:please )?"
    r"(?:(?:tell|give|show) me |do you know |i want to know )?"
)

# Each must match the whole question, so "what is the title of the book she
# mentions?" still goes to search.
_PATTERNS = {
    "title": rf"{_WHAT}(?:{_VIDEO} )?(?:title|name)(?: (?:of|for) (?:{_VIDEO}|it|this))?"
    rf"|{_WHAT}{_VIDEO}s (?:title|name)"
    rf"|what(?:s| is| was)? (?:{_VIDEO}|this)(?: is)? (?:called|titled|named)",
    "channel": rf"(?:who|(?:what|which) channel) (?:made|created|uploaded|posted|published) (?:{_VIDEO}|this|it)"
    rf"|{_WHAT}(?:youtube )?(?:channel|creator|uploader)(?: name)?(?: (?:of|for) (?:{_VIDEO}|this|it))?"
    rf"|(?:what|which) channel is (?:{_VIDEO}|this|it)(?: (?:from|on))?"
    rf"|whose video is (?:this|it)",
    "length": rf"how long is (?:{_VIDEO}|this|it)"
    rf"|{_WHAT}(?:{_VIDEO} )?(?:length|duration)(?: of (?:{_VIDEO}|this|it))?",
    "link": rf"{_WHAT}(?:{_VIDEO} )?(?:link|url)(?: (?:of|to|for) (?:{_VIDEO}|this|it))?",
}
_COMPILED = {name: re.compile(p) for name, p in _PATTERNS.items()}


def _normalise(question: str) -> str:
    text = re.sub(r"['’]", "", question.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _LEAD.sub("", text)
    return re.sub(r" (?:please|thanks|thank you)$", "", text)


def answer_from_details(question: str, info: VideoInfo) -> str | None:
    """A direct answer, or None when the question isn't about the video's details."""
    text = _normalise(question)
    for name, pattern in _COMPILED.items():
        if not pattern.fullmatch(text):
            continue
        if name == "title":
            return f"The video is titled “{info.title}”."
        if name == "channel" and info.channel:
            return f"It's from the YouTube channel {info.channel}."
        if name == "length":
            return f"About {format_time(info.duration)}, going by its last caption."
        if name == "link":
            return info.url
    return None
