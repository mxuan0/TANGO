ffmpeg -framerate $3 -start_number 0 -i $1%d.png -s 1800x1200 -c:v libx264 -profile:v high -crf 5 -pix_fmt yuv420p -threads 20 $2.mp4
