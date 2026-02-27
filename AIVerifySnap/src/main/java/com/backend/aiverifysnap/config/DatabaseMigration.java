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

        // Ensure the sequence exists for scan_id generation
        try {
            jdbcTemplate.execute(
                "CREATE SEQUENCE IF NOT EXISTS detection_history_scan_id_seq " +
                "START WITH 1 INCREMENT BY 1"
            );
            log.info("Ensured detection_history_scan_id_seq sequence exists");
        } catch (Exception e) {
            log.debug("Sequence creation skipped: {}", e.getMessage());
        }

        // Sync the sequence with the current max scan_id to avoid conflicts
        try {
            jdbcTemplate.execute(
                "SELECT setval('detection_history_scan_id_seq', " +
                "COALESCE((SELECT MAX(scan_id) FROM detection_history), 0) + 1, false)"
            );
            log.info("Synced detection_history_scan_id_seq with current max scan_id");
        } catch (Exception e) {
            log.debug("Sequence sync skipped: {}", e.getMessage());
        }
    }
}
