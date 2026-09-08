#!/bin/bash
# Fast setup check (<2 min, no GPU jobs): import every baseline's entry-point
# dependencies with the right interpreter, and exercise the queue mechanics
# with dummy chained tasks. Run after env/setup_env.sh and after ANY change
# to env/ or the queue scripts. All lines must end with OK.
set -u
cd "$(dirname "$0")/.."
source env.sh
fail=0
chk() { # name workdir interpreter import-stmt
  if (cd "$2" && "$3" -c "$4" >/dev/null 2>&1); then echo "OK  $1"
  else echo "FAIL $1  (cd $2 && $3 -c '$4')"; fail=1; fi
}
chk pacnerf PAC-NeRF "$BASELINES_PY" "import matting, mmcv, taichi, torch_scatter, cv2"
chk gic GIC "$BASELINES_PY" "import taichi, torch, trimesh, diff_gauss; from simple_knn import _C"
chk springgaus Spring-Gaus "$BASELINES_PY" "import yacs, termcolor, git, cv2, lpips, pytorch3d; from diff_gaussian_rasterization import GaussianRasterizationSettings"
if [ -x "${MASIV_PY:-}" ]; then
  chk masiv MASIV "$MASIV_PY" "import taichi, torch, kornia, warp, diff_gauss; from simple_knn import _C"
else echo "SKIP masiv (env not created)"; fail=1; fi
if [ -x NeuMA/.venv/bin/python ]; then
  chk neuma NeuMA .venv/bin/python "import warp, e3nn, torch; import diff_gaussian_rasterization"
else echo "SKIP neuma (venv not created)"; fi

# queue mechanics: two dummy tasks with && chains through a scratch queue
tmp=$(mktemp -d)
printf 'pf:a|%s|A_DONE|python -c "print(1)" && touch A_DONE\n' "$tmp" > "$tmp/tasks.txt"
( cd "$tmp" && Q="$tmp/tasks.txt" bash -c '
  line=$(head -1 "$Q"); IFS="|" read -r tag wd done cmd <<< "$line"
  cd "$wd" && bash -c "${cmd/#python /$0 }" >/dev/null 2>&1 && [ -f "$wd/$done" ]' "$BASELINES_PY" ) \
  && echo "OK  queue-exec" || { echo "FAIL queue-exec"; fail=1; }
rm -rf "$tmp"
exit $fail
