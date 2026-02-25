package com.backend.aiverifysnap.service;

import com.backend.aiverifysnap.dto.UserDto;
import com.backend.aiverifysnap.model.Users;
import com.backend.aiverifysnap.repository.UserRepository;
import jakarta.transaction.Transactional;
import org.springframework.stereotype.Service;

@Service
public class UserService {

    private final UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public UserDto getUserByName(String name) {
        Users user = userRepository.findByName(name).orElseThrow(() -> new RuntimeException("User not found!"));
        return convertToDto(user);
    }

    public Users registerUser(Users user) {
        Users usersaved = userRepository.save(user);
        return usersaved;
    }

    @Transactional
    public UserDto updateUser(String name, Users user) {
        Users existingUser = userRepository.findByName(name).orElseThrow(() -> new RuntimeException("User notfound"));
        if((user.getName() != null && !user.getName().isEmpty()) ||
                (user.getEmail() != null && !user.getEmail().isEmpty()) ||
                (user.getPasswordHash() != null && !user.getPasswordHash().isEmpty())
        ) {
            if (user.getName() != null) existingUser.setName(user.getName());
            if (user.getEmail() != null) existingUser.setEmail(user.getEmail());
            if (user.getPasswordHash() != null) existingUser.setPasswordHash(user.getPasswordHash());
            if (user.getRole() != null) existingUser.setRole(user.getRole());
        }
        Users updatedUser = userRepository.save(existingUser);
        return convertToDto(updatedUser);
    }

    @Transactional
    public void deleteUserByName(String name) {
        if(!userRepository.findByName(name).isPresent()) {
            throw new RuntimeException("User not found!");
        }
        userRepository.deleteByName(name);
    }

    private UserDto convertToDto(Users user) {
        return new UserDto(user.getId(), user.getName(), user.getEmail(), user.getRole());
    }
}
