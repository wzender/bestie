#!/bin/bash

# List of all Bootswatch themes
themes=(
  cerulean cosmo cyborg darkly flatly journal litera lumen lux
  materia minty morph pulse quartz sandstone simplex sketchy
  slate solar spacelab superhero united vapor yeti zephyr
)

# Create target directory
mkdir -p bootswatch_themes
cd bootswatch_themes || exit

# Download each theme
for theme in "${themes[@]}"; do
  url="https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/${theme}/bootstrap.min.css"
  output="${theme}.min.css"
  echo "Downloading ${theme}..."
  curl -s -o "$output" "$url"
done

wget https://codepen.io/chriddyp/pen/bWLwgP.css -O dash.css --user-agent="Mozilla/5.0"



echo "✅ All themes downloaded to $(pwd)"
