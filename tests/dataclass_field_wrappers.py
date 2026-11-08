"""A field-declaration wrapper chain shaped like NRA's own: a declaring
wrapper that branches on `MISSING` and returns through `cast`, a codec wrapper
over it, and a forwarder whose arguments no reading can follow."""

FIELD_WRAPPERS = (
    "from dataclasses import MISSING, field\n"
    "from typing import cast\n\n\n"
    "def declared(metadata, *, default=MISSING, default_factory=MISSING):\n"
    "    if default is not MISSING and default_factory is not MISSING:\n"
    "        raise TypeError('two defaults')\n"
    "    if default is not MISSING:\n"
    "        return cast(object, field(default=default, metadata=metadata))\n"
    "    if default_factory is not MISSING:\n"
    "        return cast(object, field(default_factory=default_factory, metadata=metadata))\n"
    "    return cast(object, field(metadata=metadata))\n\n\n"
    "def codec_field(codec, *, default=MISSING):\n"
    "    return declared({'codec': codec}, default=default)\n\n\n"
    "def forwarded_field(**options):\n"
    "    return field(**options)\n\n\n"
    "class PathCodec:\n    pass\n\n\n"
    "class LineCodec:\n    pass\n"
)
