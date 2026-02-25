package com.backend.aiverifysnap.repository;

import com.backend.aiverifysnap.model.Users;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserRepository extends JpaRepository<Users, Long> {

    Optional<Users> findByName(String name);

    void deleteByName(String name);
}
