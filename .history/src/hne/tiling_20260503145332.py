



def crop_and_save_tiles():
    """
    Generate & save image tiles (H&E crops)
    """






hne_img
width, height = hne_img.size
hne_img.size
***H&E pixel size:***
spot_diameter = 55 # µm
spot_diameter_fullres = scalefactors["spot_diameter_fullres"]
tissue_hires_scalef = scalefactors["tissue_hires_scalef"]

fullres_pixel_size = spot_diameter / spot_diameter_fullres
hires_pixel_size = spot_diameter / (spot_diameter_fullres * tissue_hires_scalef)
print(f"fullres pixel size is {fullres_pixel_size} µm / pixel")
print(f"hires pixel size is {hires_pixel_size} µm / pixel ≈ 9 µm/pixel ")
***Physical size of the hires H&E image:***
2000 px × 9 µm/px = 18,000 µm = 18 mm
1830 px × 9 µm/px = 16,470 µm = 16.5 mm
##### ***A quick visualization***
merged["pxl_row_in_fullres"].min()
merged["pxl_row_in_fullres"].max()
x = merged["pxl_col_in_fullres"] * tissue_hires_scalef
y = merged["pxl_row_in_fullres"] * tissue_hires_scalef

print("X coordination should be within 0 → 2000 (width):", x.min().round(2), x.max().round(2)) # cols
print("Y coordination should be within 0 → 1830 (height):", y.min().round(2), y.max().round(2)) # rows
plt.figure(figsize=(6,6))
plt.imshow(hne_img)

plt.scatter(
    x,    # x = pxl_col
    y,    # y = pxl_row
    s=6,
    c="#7FB3D5",
    alpha=0.7
)

plt.title("Spot overlay check")
plt.axis("off")
plt.show()
moving on...
# tile size on TMA scale: 0.6 mm - 2 mm

# 0.6 mm = 600 µm 
# 600 µm ÷ 9 µm/pixel ≈ 67 pixels

# 2 mm = 2000 µm
# 2000 µm ÷ 9 µm/pixel ≈ 222 pixels


TILE_SIZE = 100 # ≈ 1 mm in hires image
plt.figure(figsize=(6,6))
plt.imshow(hne_img)

for r in range(0, hne_img.height, TILE_SIZE):
    plt.axhline(r, color="black", linewidth=0.3)

for c in range(0, hne_img.width, TILE_SIZE):
    plt.axvline(c, color="black", linewidth=0.3)

plt.axis("off")
plt.show()
18 * 20
merged["x_hires"] = merged["pxl_col_in_fullres"] * tissue_hires_scalef
merged["y_hires"] = merged["pxl_row_in_fullres"] * tissue_hires_scalef
                             
merged["tile_col"] = (merged["x_hires"] // TILE_SIZE).astype(int).copy()    # tile_col = index tiles (0,1,2,3,…) vertically
merged["tile_row"] = (merged["y_hires"] // TILE_SIZE).astype(int).copy()    # tile_row = index tiles (0,1,2,3,…) horizontally

merged["tile_id"] = merged["tile_row"].astype(str) + "-" + merged["tile_col"].astype(str)
merged.head()
len(merged['tile_id'].unique())


# Subset tiles
# tile-level df
tile_df = df[['tile_id', 'tile_purity']].drop_duplicates()
plt.figure(figsize=(5, 4))
sns.histplot(tile_df['tile_purity'], bins=30, kde=True)
plt.axvline(0.2, color='red', linestyle='--', label='Threshold 0.2')
plt.title('Distribution of Bayesian Tile Purity')
plt.xlabel('Tile Purity')
plt.ylabel('Count')
plt.legend()
plt.show()
tmp = df.groupby('tile_id')['tumor_fraction'].agg(['mean','count','sum'])
tmp['tile_purity'] = (tmp['sum'] + 0.5 * 2) / (tmp['count'] + 2)

plt.figure(figsize=(5, 5))
plt.scatter(tmp['mean'], tmp['tile_purity'], alpha=0.6)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('Raw Tile Mean')
plt.ylabel('Bayesian Tile Purity (k=2)')
plt.title('Shrinkage Effect')
plt.show()
for t in [0.1, 0.2, 0.3, 0.4]:
    n_tiles = (tile_df['tile_purity'] > t).sum()
    print(f"Tiles > {t}: {n_tiles}")
tmp = df.groupby('tile_id')['tumor_fraction'].agg(['mean','count','sum'])
tmp['tile_purity'] = (tmp['sum'] + 0.5 * 2) / (tmp['count'] + 2)

plt.figure(figsize=(5, 3))
sns.scatterplot(data=tmp, x='count', y='tile_purity')
plt.xlabel('Number of Spots in Tile')
plt.ylabel('Tile Purity')
plt.title('Tile Purity vs Spot Count')
plt.show()
tumor_tiles = df[df["tile_purity"] > 0.2]
tumor_tiles["tile_id"].value_counts()

a = len(tumor_tiles["tile_id"].unique())
b = len(tumor_tiles)
print(f"# of tiles with more than 20% tumor is {a} with {b} spots.")
If threshold is too low → model sees too many non-tumor tiles → attention modie astes capacity <br> If threshold is too high → model sees only homogeneous tumor → weak supervision signal becomes too narrow → overfitting

The tile purity distro suggests 0.2 cleanly separate:
- normal + mixed background
- infiltrated tumor + pure tumor
it aligns with biological logic:
- 0.2 tumor RNA at a 55–100 µm spot → almost certainly tumor infiltration
- < 0.15 → could be small cluster or noise, best excluded
## ***Extract tiles***
df.head()
# tile-level table
tiles_stats = df.groupby(["tile_row", "tile_col"]).agg(
    tile_id=("tile_id", "first"),
    tile_purity=("tile_purity", "first"),
    n_spots=("barcode", "count")
).reset_index()

tiles_stats["n_spots"].describe()
tiles_stats
***Filter tumor tiles***
TUMOR_THRESHOLD = 0.2
MIN_SPOTS = 5

tumor_tiles = tiles_stats.query(
    "tile_purity >= @TUMOR_THRESHOLD and n_spots >= @MIN_SPOTS"
)
os.makedirs("tiles", exist_ok=True)
tumor_tiles.head()
tiles = []

for _, row in tumor_tiles.iterrows():

    tile_row = int(row["tile_row"])
    tile_col = int(row["tile_col"])

    left = tile_col * TILE_SIZE
    upper = tile_row * TILE_SIZE
    right = (tile_col + 1) * TILE_SIZE
    lower = (tile_row + 1) * TILE_SIZE

    tile_img = hne_img.crop((left, upper, right, lower))
    tiles.append(tile_img)

    tile_id = f"tile_r{tile_row}_c{tile_col}"
    tile_img.save(f"tiles/{tile_id}.png")