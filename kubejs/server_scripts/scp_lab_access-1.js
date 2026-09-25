// Keycard-controlled shutters used by the DeceasedCraft SCP laboratory prefab.
// Readers permanently open nearby iron-bar shutters in that individual facility.
//
// FIX (2026-09-25): KubeJS 6 (2001.x) event.cancel() throws an EventExit and stops
// the handler immediately. The previous version called cancel() BEFORE doing any
// work, so no message was shown and no bars were removed. cancel() is now the
// LAST statement on every path.

const SCP_LAB_READER_LEVELS = {
  'scp_additions:left_reader': 1,
  'scp_additions:left_reader_accept': 1,
  'scp_additions:left_reader_wrong': 1,
  'scp_additions:right_reader': 1,
  'scp_additions:right_reader_accept': 1,
  'scp_additions:right_reader_wrong': 1,
  'scp_additions:lv_2_left_reader': 2,
  'scp_additions:lv_2_left_reader_accept': 2,
  'scp_additions:lv_2_left_reader_wrong': 2,
  'scp_additions:lv_2_right_reader': 2,
  'scp_additions:lv_2_right_reader_accept': 2,
  'scp_additions:lv_2_right_reader_wrong': 2,
  'scp_additions:lv_3_left_reader': 3,
  'scp_additions:lv_3_left_reader_accept': 3,
  'scp_additions:lv_3_left_reader_wrong': 3,
  'scp_additions:lv_3_right_reader': 3,
  'scp_additions:lv_3_right_reader_accept': 3,
  'scp_additions:lv_3_right_reader_wrong': 3,
  'scp_additions:lv_4_left_reader': 4,
  'scp_additions:lv_4_left_reader_accept': 4,
  'scp_additions:lv_4_left_reader_wrong': 4,
  'scp_additions:lv_4_right_reader': 4,
  'scp_additions:lv_4_right_reader_accept': 4,
  'scp_additions:lv_4_right_reader_wrong': 4,
  'scp_additions:lv_5_left_reader': 5,
  'scp_additions:lv_5_left_reader_accept': 5,
  'scp_additions:lv_5_left_reader_wrong': 5,
  'scp_additions:lv_5_right_reader': 5,
  'scp_additions:lv_5_right_reader_accept': 5,
  'scp_additions:lv_5_right_reader_wrong': 5,
  'scp_additions:lv_6_left_reader': 6,
  'scp_additions:lv_6_left_reader_accept': 6,
  'scp_additions:lv_6_left_reader_wrong': 6,
  'scp_additions:lv_6_right_reader': 6,
  'scp_additions:lv_6_right_reader_accept': 6,
  'scp_additions:lv_6_right_reader_wrong': 6
}

const SCP_LAB_CARD_LEVELS = {
  'scp_additions:level_1_keycard': 1,
  'scp_additions:level_2_keycard': 2,
  'scp_additions:level_3_keycard': 3,
  'scp_additions:level_4_keycard': 4,
  'scp_additions:level_5_keycard': 5,
  'scp_additions:level_6_keycard': 6
}

const SCP_LAB_DECORATIVE_SCANNERS = {
  'scp_the_facility:keycard_scanner': true,
  'scp_the_facility:keycard_scanner_correct': true,
  'scp_the_facility:keycard_scanner_incorrect': true
}

const findNearbyReader = (level, origin, radius) => {
  let nearestReader = null
  let nearestDistance = 999

  for (let x = origin.x - radius; x <= origin.x + radius; x++) {
    for (let y = origin.y - 2; y <= origin.y + 2; y++) {
      for (let z = origin.z - radius; z <= origin.z + radius; z++) {
        const readerLevel = SCP_LAB_READER_LEVELS[level.getBlock(x, y, z).id] || 0
        if (!readerLevel) continue

        const distance = Math.abs(x - origin.x) + Math.abs(y - origin.y) + Math.abs(z - origin.z)
        if (distance < nearestDistance) {
          nearestDistance = distance
          nearestReader = { level: readerLevel, x: x, y: y, z: z }
        }
      }
    }
  }

  return nearestReader
}

BlockEvents.rightClicked(event => {
  // The event fires once per hand; only handle the main hand.
  if (String(event.hand) != 'MAIN_HAND') return

  const clicked = event.block.pos
  const cardLevel = SCP_LAB_CARD_LEVELS[event.item.id] || 0
  let reader = null

  const clickedReaderLevel = SCP_LAB_READER_LEVELS[event.block.id] || 0
  if (clickedReaderLevel) {
    reader = { level: clickedReaderLevel, x: clicked.x, y: clicked.y, z: clicked.z }
  }

  if (!reader && SCP_LAB_DECORATIVE_SCANNERS[event.block.id]) {
    reader = findNearbyReader(event.level, clicked, 8)
  }

  // Reader selection boxes extend outside their block cell, so the ray trace
  // may report the supporting wall. With a keycard held, resolve to the nearest reader.
  if (!reader && cardLevel > 0) {
    reader = findNearbyReader(event.level, clicked, 3)
  }

  if (!reader) return

  if (cardLevel < reader.level) {
    event.player.tell(`§c접근 거부: ${reader.level}등급 이상의 보안 카드가 필요합니다.`)
    event.cancel() // must be last: stops execution immediately in KubeJS 6
    return
  }

  let opened = 0

  for (let x = reader.x - 6; x <= reader.x + 6; x++) {
    for (let y = reader.y - 2; y <= reader.y + 2; y++) {
      for (let z = reader.z - 6; z <= reader.z + 6; z++) {
        if (event.level.getBlock(x, y, z).id == 'minecraft:iron_bars') {
          // Vanilla setblock sends an explicit block update (avoids client ghost bars).
          event.server.runCommandSilent(`execute in ${event.level.dimension} run setblock ${x} ${y} ${z} minecraft:air replace`)
          opened++
        }
      }
    }
  }

  if (opened > 0) {
    event.player.tell(`§a${reader.level}등급 인증 완료. 보안 셔터가 해제되었습니다.`)
    event.server.runCommandSilent(`execute in ${event.level.dimension} run playsound minecraft:block.iron_door.open master ${event.player.username} ${reader.x} ${reader.y} ${reader.z} 1 0.8`)
  } else {
    event.player.tell('§7이미 해제된 보안 구역입니다.')
  }

  // Cancel LAST so SCP Facility does not show "An Active Generator is needed".
  event.cancel()
})
