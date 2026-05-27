# Provider candidates for expanded coverage

Last checked: 2026-05-27

Scope used for this pass:

- Better quality than the existing fallbacks: SRTM 30 m for DTM/elevation and Sentinel-2 for imagery.
- Open use. Attribution requirements are acceptable.
- No account login, password, per-user token, or API key required for normal access.
- Not already implemented in this repository.
- Not just a same-country/same-dataset enhancement of an existing provider. Imagery can still be listed for a country where only DTM exists, and vice versa.

Already implemented DTM providers in this repo: SRTM, USGS 3DEP WCS, Canada HRDEM, ArcticDEM, REMA Antarctica, Austria, Baden-Wurttemberg, Bavaria, Czechia, Czech DMR5G, Denmark, England, Finland, Flanders, France, Hessen, Italy, Lithuania, Mecklenburg-Vorpommern, Lower Saxony, Netherlands AHN4, Norway, NRW, Poland, Sachsen-Anhalt, Scotland, Spain, Sweden, Switzerland, Thuringia, Wales.

Already implemented imagery providers in this repo: Sentinel-2 L2A, NAIP, Austria basemap orthophoto, France BD ORTHO, Spain PNOA, Netherlands PDOK high-resolution orthophoto, Luxembourg orthophoto, Copernicus VHR 2021, Germany NRW/Bavaria/Hessen/Lower Saxony/Thuringia DOP, Poland high-resolution orthophoto.

## Strong candidates

### New Zealand LINZ Elevation

- Type: DTM/DEM.
- Coverage: New Zealand, organized by region and survey.
- Quality: 1 m LiDAR-derived DEM and DSM Cloud Optimized GeoTIFFs.
- Access: Public AWS S3 bucket, no AWS account required: `aws s3 ls --no-sign-request s3://nz-elevation/`.
- License: CC BY 4.0, with licensor attribution from STAC collection metadata.
- Source: [New Zealand Elevation - Registry of Open Data on AWS](https://registry.opendata.aws/nz-elevation/), [LINZ access elevation data](https://www.linz.govt.nz/products-services/data/types-linz-data/elevation-data/access-elevation-data).
- Implementation notes: This is a very good fit for a STAC/COG-backed provider. Query STAC collections by bbox, read intersecting COG windows, and mosaic/reproject locally. This avoids a tile-service API key.

### New Zealand LINZ Imagery

- Type: Imagery.
- Coverage: New Zealand, organized by region and survey.
- Quality: high-resolution aerial imagery, down to 5 cm in some urban areas; also includes lower-resolution satellite/historical imagery where applicable.
- Access: Public AWS S3 bucket, no AWS account required: `aws s3 ls --no-sign-request s3://nz-imagery/`.
- License: CC BY 4.0, with licensor attribution from STAC collection metadata.
- Source: [New Zealand Imagery - Registry of Open Data on AWS](https://registry.opendata.aws/nz-imagery/), [LINZ imagery repository mirror](https://gitmemories.com/index.php/linz/imagery).
- Implementation notes: Same COG/STAC pattern as LINZ elevation. Prefer latest aerial survey intersecting the ROI; fall back by date or resolution where coverage overlaps.

### Estonia Land and Spatial Development Board DTM

- Type: DTM.
- Coverage: Estonia.
- Quality: 1 m GeoTIFF/XYZ tiles, plus country-wide 5 m, 10 m, and 25 m GeoTIFF products.
- Access: Public download page; no login indicated.
- License: Estonian open data terms with attribution.
- Source: [Download Elevation Data - Estonian Land and Spatial Development Board](https://geoportaal.maaamet.ee/eng/Maps-and-Data/Elevation-data/Download-Elevation-Data-p664.html), [Estonian key geodatasets are now Open Data](https://estgis.ee/en/news-and-announcements/estonian-key-geodatasets-are-now-open-data/).
- Implementation notes: The 1 m product appears map-sheet based, so a provider probably needs a map-sheet lookup/download step. The 5 m whole-country GeoTIFF is easier as a first implementation and still beats SRTM substantially.

### Estonia orthophotos

- Type: Imagery.
- Coverage: Estonia.
- Quality: high-resolution orthophotos since 2002, available as GeoTIFF/ECW; public WMS also exists.
- Access: Public downloads/WMS; no auth indicated.
- License: Estonian open data terms with attribution.
- Source: [Estonian key geodatasets are now Open Data](https://estgis.ee/en/news-and-announcements/estonian-key-geodatasets-are-now-open-data/), [Maa-amet OSM data/source notes](https://wiki.openstreetmap.org/wiki/Maa-amet).
- Implementation notes: If a WMS layer can return GeoTIFF or high-quality RGB, model this after the existing WMS imagery providers. If not, prefer tiled GeoTIFF downloads.

### Latvia LGIA orthophoto maps

- Type: Imagery.
- Coverage: Latvia.
- Quality: country-wide orthophoto cycles. Recent published cycles include 0.25 m/pixel color and infrared orthophotos; older cycles range 0.4-1 m.
- Access: LGIA open-data downloads and viewing formats; no auth for the listed open data downloads.
- License: CC BY 4.0.
- Source: [LGIA Open data](https://www.lgia.gov.lv/en/atvertie-dati).
- Implementation notes: Good first target for Latvia imagery. The source page exposes links per cycle; provider should prefer the most recent no-auth open cycle and document whether it is RGB or CIR.

### Latvia LGIA height data

- Type: DTM/elevation.
- Coverage: Latvia.
- Quality: 20 m digital terrain model and classified LiDAR point data with ground-point average density at least 1.5 points/m2.
- Access: LGIA open-data downloads; no auth for the listed open data downloads.
- License: CC BY 4.0.
- Source: [LGIA Open data](https://www.lgia.gov.lv/en/atvertie-dati).
- Implementation notes: The ready DTM is only 20 m, but still better than SRTM. The LiDAR points are higher quality, but would require a point-cloud-to-DTM processing path that the current provider architecture may not yet have.

### Slovenia GURS digital elevation model

- Type: DTM/DEM.
- Coverage: Slovenia.
- Quality: official digital elevation model; third-party/open DEM lists report Slovenia LiDAR DTM availability around 1 m, but the exact current product resolution should be verified from the download metadata before implementation.
- Access: Public Geodetic Data application; free data access listed.
- License: CC BY 4.0 with attribution to the Surveying and Mapping Authority of the Republic of Slovenia.
- Source: [Access to geodetic data - E-prostor](https://www.e-prostor.gov.si/en/access-to-geodetic-data/), [OpenDEM Searcher](https://opendem.info/opendemsearcher/).
- Implementation notes: Candidate is strong on license and coverage, but needs a short technical spike to identify stable direct download URLs or OGC service URLs.

### Ireland OPW/GSI open LiDAR

- Type: DTM/DSM.
- Coverage: many urban and coastal areas across Ireland, not full national coverage.
- Quality: 2 m and 5 m grid products.
- Access: Geological Survey Ireland Open Topographic Data Viewer; no auth indicated.
- License: CC BY 4.0, usable commercially and non-commercially with attribution.
- Source: [OPW press release](https://www.gov.ie/en/office-of-public-works/press-releases/opw-releases-lidar-captured-as-part-of-flood-risk-management-projects-as-open-data/), [GSI viewer item](https://www.arcgis.com/home/item.html?id=b7c4b0e763964070ad69bf8c1572c9f5).
- Implementation notes: Regional coverage means `contains()`/coverage checks need to consult the available tile/coverage index instead of advertising all of Ireland.

### Australia ELVIS elevation

- Type: DEM/DTM/elevation and bathymetry.
- Coverage: Australia, strongest in coastal, metro, flood-prone, and surveyed areas.
- Quality: source-dependent, commonly 1 m, 2 m, 5 m, 10 m, and 1-second products. Geoscience Australia also publishes 5 m drainage-catchment DEM mosaics derived from ELVIS source data.
- Access: ELVIS portal for discovery/download. No login was found for normal portal use in this pass, but the programmatic API should be verified before implementation.
- License: Open government / CC BY style source licensing depending on custodian.
- Source: [Geoscience Australia Digital Elevation Data](https://www.ga.gov.au/scientific-topics/national-location-information/digital-elevation-data), [ELVIS download guide](https://lidarvisor.com/download-elvis-australia/).
- Implementation notes: Treat as a coverage-indexed provider, not a simple country-wide raster. The 5 m catchment mosaics may be easier than individual LiDAR project products.

### Japan GSI elevation tiles

- Type: DEM/elevation.
- Coverage: Japan.
- Quality: DEM1A, DEM5A, DEM5B, DEM5C, DEM10B tile hierarchy; 1 m/5 m products are not everywhere, DEM10B is broad fallback within Japan.
- Access: Public XYZ PNG elevation tiles, no auth.
- License: GSI website content is under Japan Public Data License 1.0 unless otherwise stated.
- Source: [GSI elevation sample/program page](https://maps.gsi.go.jp/development/elevation.html), [GSI Website Terms of Use](https://www.gsi.go.jp/ENGLISH/page_e30286.html), [GSI elevation data explanation PDF](https://cyberjapandata.gsi.go.jp/help/pdf/demapi.pdf).
- Implementation notes: This is not ordinary RGB imagery. Elevation is encoded in PNG RGB values and must be decoded. A provider should request DEM1A first, then DEM5A/5B/5C/10B as fallback per tile.

### Switzerland SWISSIMAGE

- Type: Imagery.
- Coverage: Switzerland.
- Quality: official swisstopo orthophoto mosaic.
- Access: WMS/WMTS geoservices; no authorization/license required under swisstopo OGD terms.
- License: Open Government Data; free commercial use with source attribution.
- Source: [swisstopo free geodata](https://shop.swisstopo.admin.ch/en/free-geodata), [swisstopo geoservices](https://www.swisstopo.admin.ch/en/geoservices-with-swisstopo-geodata), [WMTS documentation](https://docs.geo.admin.ch/visualize-data/wmts.html).
- Implementation notes: DTM Switzerland already exists in the repo, but imagery does not. The direct WMTS layer `ch.swisstopo.swissimage` is likely the simplest path.

### Wallonia orthophoto WMS

- Type: Imagery.
- Coverage: Wallonia, Belgium.
- Quality: official orthophotography WMS.
- Access: Public WMS, free access for the public.
- License: SPW web-service terms; open/public access. Verify derivative/commercial reuse language before coding if downstream products are redistributed.
- Source: [Wallonia orthophoto WMS service listing](https://directory.spatineo.com/service/88486/), [Wallonia geoportal](https://geoportail.wallonie.be/).
- Implementation notes: Flanders DTM exists in the repo, but Wallonia imagery does not. This should fit the existing WMS imagery base class.

### Iceland 10 m DEM

- Type: DEM/elevation.
- Coverage: Iceland.
- Quality: 10 m national DEM.
- Access: Public downloads/services; no auth found.
- License: CC BY 4.0 reported by downstream registry/usage pages; verify current National Land Survey of Iceland terms before implementation.
- Source: [IslandsDEM 10 m overview](https://eleroy3.github.io/awesome-gee-community-datasets/projects/iceland_dem/), [Zenodo derivative download](https://zenodo.org/records/11453396).
- Implementation notes: Not LiDAR-class, but better than SRTM and useful national coverage. Prefer the official Landmaelingar Islands source if stable direct download URLs are available.

## Rejected or near misses

- Brussels UrbIS orthophotos: clean access/license profile, but dropped because the coverage is only one city/region.
- OpenAerialMap: dropped because coverage is patchy and unpredictable rather than a dependable country/region provider.
- Finland orthophoto WMTS: open data and good quality, but the open interface requires a user-specific API key. Source: [NLS Finland map image service](https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/map-interface-services/map-image-service-wms-wmts).
- Denmark Dataforsyningen orthophotos/DHM: good national data, but web services require a user/token. The repo already has a Denmark DTM provider with token settings, so it does not match the no-auth requirement.
- Sweden Lantmateriet elevation/orthophoto services: repo already has Sweden DTM and it requires credentials; imagery access also appears credentialed or not cleanly no-auth.
- Norway elevation: repo already has Norway DTM. Norwegian topographic services are open, but they are not a better imagery replacement unless a specific no-auth orthophoto source is verified.
- Switzerland DTM/elevation: repo already has Switzerland DTM. SWISSIMAGE is still listed above because imagery is not implemented.
- USGS 3DEP/S1M: repo already has a 1 m USGS WCS provider using the 3DEP elevation service. Newer S1M products may improve consistency, but this is an enhancement to the existing USGS provider rather than a new provider candidate.
- OpenDTM Germany: removed as a strong candidate because it overlaps many existing German DTM providers in this repo. It may still be useful later as a source-finder for missing German states, but it should not be treated as a clean new provider without splitting out only non-implemented states.
- Italy regional 1 m LiDAR DTMs: removed as a candidate because the repo already has `ItalyProvider` for Italy DTM. These regional LiDAR sources may be a future quality upgrade or supplemental regional provider, but they are not a clean non-duplicate entry under this pass.
