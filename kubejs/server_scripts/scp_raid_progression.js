// DeceasedCraft SCP raid progression
//
// Outside world: only Level 1 access cards can be found.
// Facility loot tables (added in the next phase) own Levels 2-6.
// This keeps the loop as: city scavenging -> lab entry -> deeper clearance.

ServerEvents.recipes(event => {
  // SCP Additions ships cheap crafting recipes for every clearance level.
  // Remove them so cards remain raid progression items instead of craftables.
  for (let level = 1; level <= 6; level++) {
    event.remove({ id: `scp_additions:level_${level}_recipe` })
  }
})

ServerEvents.chestLootTables(event => {
  const addEntryCard = (tableId, cardWeight, emptyWeight) => {
    event.modify(tableId, table => {
      table.addPool(pool => {
        pool.setUniformRolls(1, 1)
        pool.addItem('scp_additions:level_1_keycard', cardWeight)
        pool.addEmpty(emptyWeight)
      })
    })
  }

  // Police facilities: the most reliable external source of a lab entry card.
  const policeSecureLoot = [
    'deceasedcraft:building_policeoffice1/base1/safe',
    'deceasedcraft:building_policeoffice1/base2/iron_locker',
    'deceasedcraft:building_policeoffice1/base2/large_shelves_fill',
    'deceasedcraft:building_policeoffice1/base2/lootr_chest',
    'deceasedcraft:building_policeoffice1/floor3/iron_locker',
    'deceasedcraft:multi_policeapartment_0/floor2/lootr_chest',
    'deceasedcraft:multi_policeapartment_0/ground/iron_locker',
    'deceasedcraft:multi_policeapartment_0/ground/lootr_chest',
    'deceasedcraft:multi_policeapartment_1/base1/metal_shelvels_two',
    'deceasedcraft:multi_policeapartment_2/floor2/lootr_chest',
    'deceasedcraft:multi_policeapartment_2/top/iron_locker',
    'deceasedcraft:multi_policeapartment_2/top/lootr_chest',
    'deceasedcraft:multi_policestation_0/ground/iron_locker',
    'deceasedcraft:multi_policestation_1/base1/iron_locker',
    'deceasedcraft:multi_policestation_1/base1/metelshelvesfour',
    'deceasedcraft:multi_policestation_1/base1/safe',
    'deceasedcraft:street_highrise_office_all/police/lootr_chest'
  ]

  // Roughly 8% per selected police container.
  policeSecureLoot.forEach(tableId => addEntryCard(tableId, 8, 92))

  // Hospitals can plausibly hold transferred staff credentials, but less often.
  const hospitalSecureLoot = [
    'deceasedcraft:multi_hospital_3/base5/iron_locker',
    'deceasedcraft:multi_hospital_4/top/safe',
    'deceasedcraft:multi_hospital_5/ground/medical_storage',
    'deceasedcraft:multi_hospitaldamaged_0/top/safe',
    'deceasedcraft:multi_hospitaldamaged_3/ground/medical_storage',
    'deceasedcraft:multi_hospitaldamaged_5/ground/medical_storage',
    'deceasedcraft:building_policeoffice1/ground/medical_storage'
  ]

  // Roughly 4% per selected hospital container.
  hospitalSecureLoot.forEach(tableId => addEntryCard(tableId, 4, 96))

  // Corporate safes and dedicated loot chests are the rare fallback route.
  const officeSecureLoot = [
    'deceasedcraft:building_office1b/base1/lootr_chest',
    'deceasedcraft:building_office1b/base1/safe',
    'deceasedcraft:building_office2a/floor2/lootr_chest',
    'deceasedcraft:building_office2a/top/safe',
    'deceasedcraft:building_office3a/floor3/lootr_chest',
    'deceasedcraft:building_office3a/floor4/lootr_chest',
    'deceasedcraft:building_office3a/top/iron_locker',
    'deceasedcraft:building_office3b/top/safe',
    'deceasedcraft:building_office3c/top/safe',
    'deceasedcraft:building_office5/floor4/safe',
    'deceasedcraft:building_officebank/base2/safe'
  ]

  // Roughly 2% per selected office container.
  officeSecureLoot.forEach(tableId => addEntryCard(tableId, 2, 98))

  // Facility-specific loot. These tables are referenced by the SCP lab prefab.
  event.addChest('deceasedcraft:scp_lab/entry_cache', table => {
    table.addPool(pool => {
      pool.setUniformRolls(3, 5)
      pool.addItem('minecraft:torch', 20)
      pool.addItem('minecraft:bread', 15)
      pool.addItem('minecraft:iron_ingot', 12)
      pool.addItem('minecraft:redstone', 12)
      pool.addItem('minecraft:paper', 10)
      pool.addItem('minecraft:golden_carrot', 4)
    })
  })

  event.addChest('deceasedcraft:scp_lab/security_office', table => {
    table.addPool(pool => {
      pool.setUniformRolls(1, 1)
      pool.addItem('scp_additions:level_2_keycard', 35)
      pool.addEmpty(65)
    })
    table.addPool(pool => {
      pool.setUniformRolls(3, 5)
      pool.addItem('minecraft:iron_ingot', 18)
      pool.addItem('minecraft:redstone', 16)
      pool.addItem('minecraft:gold_ingot', 8)
      pool.addItem('minecraft:compass', 5)
      pool.addItem('scp_additions:security_credentials', 4)
      pool.addItem('deceasedcraft:research_paper_1', 3)
    })
  })

  event.addChest('deceasedcraft:scp_lab/archive', table => {
    table.addPool(pool => {
      pool.setUniformRolls(1, 1)
      pool.addItem('scp_additions:level_3_keycard', 20)
      pool.addEmpty(80)
    })
    table.addPool(pool => {
      pool.setUniformRolls(2, 4)
      pool.addItem('deceasedcraft:research_paper_1', 8)
      pool.addItem('deceasedcraft:research_paper_2', 6)
      pool.addItem('deceasedcraft:research_paper_3', 4)
      pool.addItem('minecraft:paper', 18)
      pool.addItem('minecraft:book', 10)
      pool.addItem('minecraft:lapis_lazuli', 6)
    })
  })

  event.addChest('deceasedcraft:scp_lab/containment', table => {
    table.addPool(pool => {
      pool.setUniformRolls(2, 3)
      pool.addItem('scp_additions:security_credentials', 10)
      pool.addItem('deceasedcraft:research_paper_4', 5)
      pool.addItem('deceasedcraft:research_paper_5', 3)
      pool.addItem('minecraft:emerald', 8)
      pool.addItem('minecraft:diamond', 2)
      pool.addEmpty(15)
    })
  })

  // Deep-site progression. Each wing has two credential caches, so a failed
  // roll means extracting and raiding another site instead of crafting past it.
  event.addChest('deceasedcraft:scp_lab/clearance_4', table => {
    table.addPool(pool => {
      pool.setUniformRolls(1, 1)
      pool.addItem('scp_additions:level_4_keycard', 50)
      pool.addEmpty(50)
    })
    table.addPool(pool => {
      pool.setUniformRolls(2, 4)
      pool.addItem('deceasedcraft:research_paper_2', 8)
      pool.addItem('deceasedcraft:research_paper_3', 5)
      pool.addItem('minecraft:gold_ingot', 8)
      pool.addItem('minecraft:redstone', 12)
    })
  })

  event.addChest('deceasedcraft:scp_lab/clearance_5', table => {
    table.addPool(pool => {
      pool.setUniformRolls(1, 1)
      pool.addItem('scp_additions:level_5_keycard', 40)
      pool.addEmpty(60)
    })
    table.addPool(pool => {
      pool.setUniformRolls(2, 4)
      pool.addItem('deceasedcraft:research_paper_3', 7)
      pool.addItem('deceasedcraft:research_paper_4', 4)
      pool.addItem('minecraft:emerald', 7)
      pool.addItem('minecraft:lapis_lazuli', 10)
    })
  })

  event.addChest('deceasedcraft:scp_lab/clearance_6', table => {
    table.addPool(pool => {
      pool.setUniformRolls(1, 1)
      pool.addItem('scp_additions:level_6_keycard', 30)
      pool.addEmpty(70)
    })
    table.addPool(pool => {
      pool.setUniformRolls(2, 4)
      pool.addItem('deceasedcraft:research_paper_4', 7)
      pool.addItem('deceasedcraft:research_paper_5', 4)
      pool.addItem('minecraft:diamond', 2)
      pool.addItem('scp_additions:security_credentials', 8)
    })
  })

  event.addChest('deceasedcraft:scp_lab/deep_vault', table => {
    table.addPool(pool => {
      pool.setUniformRolls(4, 6)
      pool.addItem('deceasedcraft:research_paper_4', 9)
      pool.addItem('deceasedcraft:research_paper_5', 7)
      pool.addItem('scp_additions:security_credentials', 10)
      pool.addItem('minecraft:diamond', 4)
      pool.addItem('minecraft:emerald', 8)
      pool.addItem('minecraft:netherite_scrap', 1)
      pool.addEmpty(10)
    })
  })
})
