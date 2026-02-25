package com.backend.aiverifysnap.config;

import javax.sql.DataSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

@Component
public class DatabaseMigration {

    private static final Logger log = LoggerFactory.getLogger(DatabaseMigration.class);
    private final JdbcTemplate jdbcTemplate;

    public DatabaseMigration(DataSource dataSource) {
        this.jdbcTemplate = new JdbcTemplate(dataSource);
    }

    @EventListener(ApplicationReadyEvent.class)
    public void migrate() {
        try {
            jdbcTemplate.execute(
                "ALTER TABLE detection_history ALTER COLUMN analysis_metadata TYPE TEXT"
            );
            log.info("Successfully altered analysis_metadata column to TEXT");
        } catch (Exception e) {
            // Column may already be TEXT or table may not exist yet
            log.debug("Column migration skipped: {}", e.getMessage());
        }
    }
}
