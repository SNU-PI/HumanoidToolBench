/* Copy rigid native visualization scenes without rebuilding physical state. */
#include <stddef.h>
#include <string.h>
#include <mujoco/mjvisualize.h>

#define SCENE_POINTERS(X) \
  X(geoms) X(geomorder) \
  X(flexedgeadr) X(flexedgenum) X(flexvertadr) X(flexvertnum) \
  X(flexfaceadr) X(flexfacenum) X(flexfaceused) X(flexedge) \
  X(flexvert) X(flexface) X(flexnormal) X(flextexcoord) \
  X(skinfacenum) X(skinvertadr) X(skinvertnum) X(skinvert) X(skinnormal)

size_t theta_scene_flags_offset(void) {
  return offsetof(mjvScene, flags);
}

size_t theta_scene_size(const mjvScene* scene) {
  if (scene->nflex || scene->nskin || scene->ngeom < 0 || scene->ngeom > scene->maxgeom) {
    return 0;
  }
  return sizeof(mjvScene) + scene->ngeom * (sizeof(mjvGeom) + sizeof(int));
}

int theta_pack_scene(const mjvScene* scene, void* output, size_t size) {
  size_t required = theta_scene_size(scene);
  if (!required || size != required) return -1;
  mjvScene header = *scene;
#define CLEAR(name) header.name = NULL;
  SCENE_POINTERS(CLEAR)
#undef CLEAR
  char* bytes = output;
  memcpy(bytes, &header, sizeof(header));
  bytes += sizeof(header);
  memcpy(bytes, scene->geoms, scene->ngeom * sizeof(mjvGeom));
  bytes += scene->ngeom * sizeof(mjvGeom);
  memcpy(bytes, scene->geomorder, scene->ngeom * sizeof(int));
  return 0;
}

int theta_unpack_scene(mjvScene* scene, const void* input, size_t size) {
  if (size < sizeof(mjvScene)) return -1;
  mjvScene header;
  memcpy(&header, input, sizeof(header));
  if (header.nflex || header.nskin || header.ngeom < 0 || header.ngeom > scene->maxgeom ||
      size != sizeof(mjvScene) + header.ngeom * (sizeof(mjvGeom) + sizeof(int))) return -1;
  /* Keep every destination allocation owned by its original Python scene. */
#define RESTORE(name) header.name = scene->name;
  SCENE_POINTERS(RESTORE)
#undef RESTORE
  header.maxgeom = scene->maxgeom;
  *scene = header;
  const char* bytes = (const char*)input + sizeof(header);
  memcpy(scene->geoms, bytes, scene->ngeom * sizeof(mjvGeom));
  bytes += scene->ngeom * sizeof(mjvGeom);
  memcpy(scene->geomorder, bytes, scene->ngeom * sizeof(int));
  return 0;
}
