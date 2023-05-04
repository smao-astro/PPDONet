import base64
from io import BytesIO


def mpl_to_uri(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return "data:image/png;base64,{}".format(
        base64.b64encode(buf.getvalue()).decode("utf-8")
    )
