set -o allexport
source .env set

curl -s -X POST \
-d '{"destination": "platform-kit/xtts", "input": {"upload_file": "https://replicate.delivery/pbxt/JsXUwO8tw66Do3A20soFB1m3iTw3IxCPIDDsM11F4g549wPv/output%20(46).wav"}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  https://api.replicate.com/v1/models/platform-kit/xtts/versions/$REPLICATE_MODEL_VERSION/trainings