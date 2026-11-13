# Cerveza — La taberna de El Sabio

Guion visual para el tema **“Cerveza”** de El Sabio. El vídeo es una pieza animada de fantasía oscura y humor heavy metal: un bosque brumoso, una taberna dentro de un árbol colosal, una banda formada por el grupo y mucha cerveza.

## Ficha técnica

| Campo | Valor |
|---|---|
| Canción | `Cerveza.mp3` (3:16) |
| Formato horizontal | 832 × 480, 16 fps |
| Formato vertical | 480 × 832, 16 fps, mediante `--orientation vertical` |
| Duración | 49 planos de ~4 s, escalados automáticamente a los 3:16 de la canción |
| Imagen | `Qwen/Qwen-Image` y `Qwen/Qwen-Image-Edit` |
| Vídeo | `Wan-AI/Wan2.2-I2V-A14B-Diffusers`, siempre con imagen inicial y final |
| Subtítulos | `subtitles.json`; `subtitles_vertical.json` en vertical |
| Guion ejecutable único | `screenplay.json` |
| Referencias | `input/characters.json` y las imágenes de `input/` |

## Sinopsis

Un monje se pierde en un bosque encantado. Tras horas de caminar encuentra una taberna construida dentro de un árbol colosal. Dentro, El Sabio está dando un concierto: Prima canta, Saul y Rubo tocan las guitarras, Pedro el bajo, Axel la batería y Patricia el violín. El monje conoce al tabernero y al druida, aprende los secretos de la cerveza —rubia, de trigo y negra— y recibe la receta. Después de una noche de música y celebración, vuelve a su monasterio para compartir el hallazgo.

## Personajes

Los nombres y los roles proceden de `input/characters.json`. Para no obligar al editor a rediseñar a nadie, la apariencia se define únicamente mediante cada imagen de referencia, sin descripciones físicas contradictorias en los prompts. Las imágenes son anclas inmutables: no deben cambiar la cara, la edad, el peinado, el vestuario ni las proporciones de una escena a otra.

- **Monk**: protagonista y viajero que descubre la taberna; conserva siempre el hábito de su referencia.
- **Axel**: batería del grupo.
- **BarTender**: camarero, responsable de servir las jarras y custodiar la receta.
- **Druid**: maestro herbalista y cervecero.
- **Patricia**: violinista del grupo.
- **Pedro**: bajista del grupo.
- **Prima**: vocalista principal.
- **Rubo**: segundo guitarrista.
- **Saul**: guitarrista principal.

La clientela y los demás monjes son personajes de fondo y nunca deben sustituir, duplicar ni mezclar las identidades del reparto principal.

## Estética y reglas visuales

- Animación 3D estilizada de fantasía oscura, con texturas detalladas y caras reconocibles; nada de rediseños fotorrealistas del reparto.
- Paleta: verdes profundos del bosque; ámbar y naranja del fuego; dorado, tostado y negro de la cerveza; rojos y ámbar en el escenario.
- La cerveza solo puede ser rubia, tostada o negra. Nunca verde ni con reflejos verdes.
- Cada plano define `prompt` para la imagen inicial, `motion_prompt` para el movimiento y `end_prompt` para la imagen final. La cámara, el fondo, la iluminación y las identidades permanecen constantes salvo el cambio exigido por la acción.
- Monk conserva siempre su hábito marrón; no aparece un cambio de vestuario a mitad del vídeo.
- Nada de texto legible en pantalla: los carteles y pergaminos usan pictogramas de espigas, lúpulo y jarras.

## Estructura musical

| Acto | Tiempo | Planos | Acción |
|---|---|---|---|
| 1. El bosque y la taberna | 0:00–0:28 | 1–7 | Monk se pierde, descubre la taberna, ve a la banda y recibe su primera jarra. |
| 2. La primera jarra | 0:28–1:04 | 8–16 | Prueba la cerveza, conoce a los músicos y brinda con sus nuevos amigos. |
| 3. La noche se descontrola | 1:04–1:28 | 17–22 | Mosh pit, chupito y llegada de Druid, al que Monk invita a entrar. |
| 4. La receta | 1:28–1:48 | 23–27 | BarTender enseña el pergamino; Monk, Druid y Prima preparan el mosto. |
| 5. El brindis | 1:48–2:08 | 28–32 | Monk explica la receta y todos brindan bajo un géiser dorado. |
| 6. Concierto | 2:08–2:28 | 33–37 | Actuación del grupo completo y baile de Monk con sus nuevos amigos. |
| 7. Solos y caos | 2:28–2:56 | 38–44 | Solos de Saul y Axel, crowd-surfing, tejado abierto a la luna y cascada de cerveza. |
| 8. Vuelta a casa | 2:56–3:16 | 45–49 | Monk se despide, comparte la cerveza en el monasterio y brinda con todo el grupo. |

## Generación

Plano horizontal completo:

```bash
scripts/create_cerveza_video.sh
```

El mismo guion en vertical:

```bash
scripts/create_cerveza_vertical.sh
```

También se puede invocar directamente:

```bash
python scripts/create_video_from_character_screenplay.py \
  --screenplay examples/largo_viaje/el_sabio/cerveza/screenplay.json \
  --output output/cerveza_group \
  --orientation horizontal
```

Para vertical, cambia a `--orientation vertical --subtitles subtitles_vertical.json`. Los parámetros `--width` y `--height` permiten sobrescribir las dimensiones sin crear otro guion.

Usa `--dry-run` para revisar el reparto, las 49 escenas y el número de fotogramas sin cargar modelos. Las salidas se reutilizan para reanudar trabajos; después de cambiar referencias o prompts de una escena, utiliza un directorio `--output` nuevo para no conservar imágenes antiguas.
